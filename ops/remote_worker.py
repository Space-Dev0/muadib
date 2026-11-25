"""Optional SSH-owned worker with bounded lifetime and no PID-based killing.

The remote Python process runs the worker itself. Its watchdog exits that same
process on SSH stdin EOF or lease expiry; it never signals a different process.
SSH authentication and host verification remain the user's responsibility.
"""
from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import threading
import time
import uuid
import warnings

_PROGRAM = r'''
import base64, hashlib, json, os, runpy, socket, sys, threading
p = json.loads(base64.b64decode(PAYLOAD).decode('utf-8'))
marker = p['token']
def stop(reason):
    try:
        print('MUADIB_STOP_' + marker + '_' + reason, flush=True)
    finally:
        os._exit(0)
def watch_stdin():
    try:
        if os.name == 'nt':
            # A blocking read in a background thread can deadlock Windows
            # NumPy native-module initialization. Poll the SSH pipe instead.
            import ctypes, msvcrt, time
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.PeekNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                            wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
            kernel.PeekNamedPipe.restype = wintypes.BOOL
            handle = wintypes.HANDLE(msvcrt.get_osfhandle(sys.stdin.fileno()))
            while True:
                available = wintypes.DWORD()
                if not kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
                    break
                # The launcher never sends data; unexpected input closes the lease.
                if available.value:
                    break
                time.sleep(.2)
        else:
            while sys.stdin.buffer.read(4096):
                pass
    finally:
        stop('EOF')
t = threading.Thread(target=watch_stdin, daemon=True)
t.start()
timer = threading.Timer(p['lease_seconds'], stop, args=('LEASE',))
timer.daemon = True
timer.start()
os.chdir(p['cwd'])
with open(p['checkpoint'], 'rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
if digest != p['checkpoint_sha256']:
    raise SystemExit('checkpoint SHA-256 mismatch')
original_listen = socket.socket.listen
def listen_and_announce(sock, *args, **kwargs):
    result = original_listen(sock, *args, **kwargs)
    addr = sock.getsockname()
    if isinstance(addr, tuple) and addr[:2] == ('127.0.0.1', p['port']):
        with open(p['checkpoint'], 'rb') as checkpoint_file:
            if hashlib.file_digest(checkpoint_file, 'sha256').hexdigest() != digest:
                raise RuntimeError('checkpoint changed before worker readiness')
        print('MUADIB_READY_' + marker + '_' + digest, flush=True)
    return result
socket.socket.listen = listen_and_announce
sys.argv = ['ops.distributed_inference', 'worker', '--checkpoint', p['checkpoint'],
            '--format', p['format'], '--device', p['device'], '--port', str(p['port'])]
runpy.run_module('ops.distributed_inference', run_name='__main__')
'''


def _startup_diagnostic(line: str) -> str | None:
    """Classify a few known errors without disclosing untrusted remote output.

    The SSH/worker stream can include encoded launch commands, ownership
    markers, paths, or environment data. Never expose arbitrary lines from it.
    """
    checks = (
        ('REMOTE HOST IDENTIFICATION HAS CHANGED', 'SSH host-key verification failed'),
        ('Host key verification failed', 'SSH host-key verification failed'),
        ('Permission denied (publickey', 'SSH public-key authentication failed'),
        ('checkpoint SHA-256 mismatch', 'checkpoint SHA-256 mismatch'),
        ('checkpoint changed before worker readiness', 'checkpoint changed before worker readiness'),
        ('Address already in use', 'remote worker port is already in use'),
        ('Only one usage of each socket address', 'remote worker port is already in use'),
        ('ModuleNotFoundError: No module named', 'remote Python dependency is missing'),
        ('No such file or directory', 'remote executable, working directory, or checkpoint could not be found'),
    )
    return next((message for needle, message in checks if needle in line), None)


def validate_worker_config(config: object) -> dict:
    required = {'platform', 'python', 'cwd', 'checkpoint', 'format', 'device', 'lease_seconds'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('start_worker requires platform, python, cwd, checkpoint, format, device, lease_seconds')
    platform = config['platform']
    executable = config['python']
    if platform not in ('linux', 'windows') or not isinstance(executable, str):
        raise ValueError('invalid remote platform or Python executable')
    # Executable is the only user text entering the shell command. Restrict it
    # to paths with no shell metacharacters; other fields are base64 JSON.
    pattern = r'[A-Za-z0-9_./-]+' if platform == 'linux' else r'[A-Za-z0-9 _.:\\/-]+'
    if (len(executable) > 512 or not re.fullmatch(pattern, executable)
            or not executable.strip() or executable.startswith('-')):
        raise ValueError('Python executable contains unsupported shell characters')
    for name in ('cwd', 'checkpoint'):
        if not isinstance(config[name], str) or not config[name] or '\x00' in config[name] or len(config[name]) > 4096:
            raise ValueError(f'{name} must be a nonempty path')
    if config['format'] not in ('warmstart', 'probe') or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('unsupported checkpoint format or device')
    if type(config['lease_seconds']) is not int or not 30 <= config['lease_seconds'] <= 86400:
        raise ValueError('worker lease must be an integer from 30 to 86400 seconds')
    return config


def _command(config: dict, checkpoint_sha256: str, port: int, token: str) -> str:
    payload = {**config, 'checkpoint_sha256': checkpoint_sha256, 'port': port, 'token': token}
    raw = base64.b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode('ascii')
    program = 'PAYLOAD=' + repr(raw) + '\n' + _PROGRAM
    encoded = base64.b64encode(program.encode()).decode('ascii')
    executable = shlex.quote(config['python']) if config['platform'] == 'linux' else '"' + config['python'] + '"'
    command = executable + ' -u -c "exec(__import__(\'base64\').b64decode(\'' + encoded + '\'))"'
    # The Windows OpenSSH default shell is often cmd.exe (8191-character
    # command limit). Fail before starting SSH rather than silently truncating.
    if config['platform'] == 'windows' and len(command) > 7000:
        raise ValueError('encoded Windows worker command is too long')
    return command


class OwnedWorker:
    def __init__(self, tunnel: dict, worker: dict):
        config = validate_worker_config(tunnel['start_worker'])
        self.token = uuid.uuid4().hex
        self.expected = f"MUADIB_READY_{self.token}_{worker['checkpoint_sha256']}"
        self.stop_prefix = f'MUADIB_STOP_{self.token}_'
        self._condition = threading.Condition()
        self._ready = False
        self._stop_seen = False
        self._reader_finished = False
        self._diagnostic = None
        self.cleanup_confirmed = False
        args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3']
        if 'control_path' in tunnel:
            args += ['-S', tunnel['control_path']]
        else:
            args += ['-o', 'ControlMaster=no', '-o', 'ControlPath=none']
        args += ['-p', str(tunnel['ssh_port']), tunnel['host'],
                 _command(config, worker['checkpoint_sha256'], tunnel['remote_port'], self.token)]
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, bufsize=0)
        try:
            self.reader = threading.Thread(target=self._read_lines, daemon=True)
            self.reader.start()
        except BaseException:
            # ExitStack has not yet received this object, so it cannot clean up
            # the child if thread startup fails. Close only the SSH process we
            # created; remote termination remains unconfirmed without a reader.
            self._abort_reader_start()
            raise

    def _abort_reader_start(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except OSError:
            pass
        try:
            if self.process.poll() is None:
                try:
                    self.process.terminate()
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        self.process.kill()
                    except OSError:
                        pass
                    self.process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            # A closed child pipe is safe to close when no reader was started.
            try:
                if self.process.stdout:
                    self.process.stdout.close()
            except OSError:
                pass
        # This warning must not replace the original thread-start exception,
        # even when a caller has configured warnings as errors.
        try:
            warnings.warn('Remote worker cleanup unconfirmed after reader startup failure', RuntimeWarning)
        except RuntimeWarning:
            pass

    def _read_lines(self) -> None:
        try:
            while True:
                raw = self.process.stdout.readline(8192)
                if not raw:
                    break
                line = raw.decode('utf-8', errors='replace').strip()
                # Keep only constant-size control state, not arbitrary remote
                # output. READY/STOP cannot be dropped under log backpressure.
                with self._condition:
                    if line == self.expected:
                        self._ready = True
                    elif line.startswith(self.stop_prefix):
                        self._stop_seen = True
                    else:
                        self._diagnostic = _startup_diagnostic(line) or self._diagnostic
                    self._condition.notify_all()
        finally:
            # This thread owns the read end. Closing it here avoids leaking a
            # descriptor while avoiding a cross-thread close during read().
            try:
                self.process.stdout.close()
            except OSError:
                pass
            with self._condition:
                self._reader_finished = True
                self._condition.notify_all()

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._ready:
                    return
                suffix = f': {self._diagnostic}' if self._diagnostic else ''
                if self._reader_finished:
                    raise RuntimeError('remote owned worker exited before readiness' + suffix)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('remote owned worker did not become ready before deadline' + suffix)
                self._condition.wait(timeout=remaining)

    def close(self) -> None:
        forced = False
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            try:
                if self.process.poll() is None:
                    forced = True
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                forced = True
            # Only join after the SSH child has exited; otherwise a remote pipe
            # read can block. The reader closes stdout itself when it finishes.
            if self.process.poll() is not None:
                self.reader.join(timeout=1)
            with self._condition:
                self.cleanup_confirmed = (not forced and self._stop_seen and self._reader_finished
                                          and self.process.returncode == 0)
        if not self.cleanup_confirmed:
            warnings.warn('Remote worker cleanup unconfirmed; its bounded lease will expire independently', RuntimeWarning)
