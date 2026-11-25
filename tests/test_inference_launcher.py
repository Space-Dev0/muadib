import unittest
import socket
import subprocess
import struct
import threading
import time
from pathlib import Path
from unittest.mock import patch

from ops import inference_launcher as launcher

SHA = 'a' * 64
PLAN = {'schema_version': 1, 'jobs': [{'id': 'one', 'stratum': 'x', 'checkpoint_sha256': SHA,
                                      'color': 'red', 'seed': 1}]}
WORKERS = [{'id': 'node', 'port': 50441, 'checkpoint_sha256': SHA}]
TUNNELS = {'tunnels': [{'worker_id': 'node', 'host': 'user@laptop', 'ssh_port': 22, 'remote_port': 50500}]}


class FakeProcess:
    def __init__(self): self.stopped = False
    def poll(self): return 0 if self.stopped else None
    def terminate(self): self.stopped = True
    def wait(self, timeout): return 0


class LauncherTests(unittest.TestCase):
    def test_unconfirmed_tunnel_cleanup_warns_instead_of_masking_run_result(self):
        class UnreapableProcess:
            def __init__(self):
                self.killed = False
            def poll(self):
                return None
            def terminate(self):
                pass
            def kill(self):
                self.killed = True
            def wait(self, timeout):
                raise subprocess.TimeoutExpired('ssh', timeout)

        process = UnreapableProcess()
        with self.assertWarnsRegex(RuntimeWarning, 'SSH tunnel cleanup unconfirmed'):
            launcher._stop_owned(process)
        self.assertTrue(process.killed)

    def test_tunnel_owns_only_its_process_and_invokes_frozen_plan(self):
        proc = FakeProcess()
        with patch.object(launcher, '_available', return_value=False), \
             patch.object(launcher, '_worker_ready', return_value=True), \
             patch.object(launcher.time, 'sleep'), \
             patch.object(launcher.subprocess, 'Popen', return_value=proc) as spawn, \
             patch.object(launcher.adaptive_inference, 'run', return_value={'missing': 0}) as run:
            result = launcher.launch(PLAN, WORKERS, TUNNELS, Path('run'))
        self.assertEqual(result['missing'], 0)
        self.assertTrue(proc.stopped)
        args = spawn.call_args.args[0]
        self.assertIn('BatchMode=yes', args)
        self.assertIn('ExitOnForwardFailure=yes', args)
        self.assertIn('ControlMaster=no', args)
        self.assertIn('ControlPath=none', args)
        self.assertIn('127.0.0.1:50441:127.0.0.1:50500', args)
        self.assertNotIn('StrictHostKeyChecking=no', args)
        self.assertIs(run.call_args.args[0], PLAN)

    def test_existing_control_socket_uses_forward_and_cancel(self):
        config = {'tunnels': [dict(TUNNELS['tunnels'][0], control_path='/tmp/control.sock')]}
        with patch.object(Path, 'is_socket', return_value=True), \
             patch.object(Path, 'is_symlink', return_value=False), \
             patch.object(launcher, '_available', return_value=False), \
             patch.object(launcher, '_worker_ready', return_value=True), \
             patch.object(launcher.subprocess, 'Popen') as spawn, \
             patch.object(launcher.subprocess, 'run') as command, \
             patch.object(launcher.adaptive_inference, 'run', return_value={'missing': 0}):
            command.return_value.returncode = 0
            launcher.launch(PLAN, WORKERS, config, Path('run'))
        spawn.assert_not_called()
        calls = [call.args[0] for call in command.call_args_list]
        suffix = ['-L', '127.0.0.1:50441:127.0.0.1:50500', '-p', '22', 'user@laptop']
        self.assertEqual(calls, [['ssh', '-S', '/tmp/control.sock', '-O', 'forward'] + suffix,
                                 ['ssh', '-S', '/tmp/control.sock', '-O', 'cancel'] + suffix])

    def test_failed_control_forward_is_not_cancelled(self):
        config = {'tunnels': [dict(TUNNELS['tunnels'][0], control_path='/tmp/control.sock')]}
        with patch.object(Path, 'is_socket', return_value=True), \
             patch.object(Path, 'is_symlink', return_value=False), \
             patch.object(launcher, '_available', return_value=False), \
             patch.object(launcher.subprocess, 'run') as command:
            command.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'forward.*failed'):
                launcher.launch(PLAN, WORKERS, config, Path('run'))
        self.assertEqual(command.call_count, 1)

    def test_timed_out_control_forward_is_not_cancelled_when_ownership_is_uncertain(self):
        config = {'tunnels': [dict(TUNNELS['tunnels'][0], control_path='/tmp/control.sock')]}
        with patch.object(Path, 'is_socket', return_value=True), \
             patch.object(Path, 'is_symlink', return_value=False), \
             patch.object(launcher, '_available', return_value=False), \
             patch.object(launcher.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ssh', 1)) as command:
            with self.assertRaisesRegex(RuntimeError, 'ownership is uncertain'):
                launcher.launch(PLAN, WORKERS, config, Path('run'))
        self.assertEqual(command.call_count, 1)

    def test_control_socket_path_is_validated(self):
        for value in ('relative.sock', '/missing.sock', 'bad\x00path', 9):
            config = {'tunnels': [dict(TUNNELS['tunnels'][0], control_path=value)]}
            with self.subTest(value=value), self.assertRaises(ValueError):
                launcher.validate_tunnels(config, WORKERS)

    def test_cleanup_on_worker_unready(self):
        proc = FakeProcess()
        with patch.object(launcher, '_available', return_value=False), \
             patch.object(launcher, '_worker_ready', return_value=False), \
             patch.object(launcher.time, 'sleep'), \
             patch.object(launcher.time, 'monotonic', side_effect=[0, 2]), \
             patch.object(launcher.subprocess, 'Popen', return_value=proc), \
             patch.object(launcher.adaptive_inference, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                launcher.launch(PLAN, WORKERS, TUNNELS, Path('run'), startup_timeout=1)
        self.assertTrue(proc.stopped)
        run.assert_not_called()

    def test_rejects_unsafe_host_and_duplicate_workers(self):
        config = {'tunnels': [dict(TUNNELS['tunnels'][0], host='-oProxyCommand=evil')]}
        with self.assertRaises(ValueError):
            launcher.validate_tunnels(config, WORKERS)
        with self.assertRaises(ValueError):
            launcher.validate_tunnels({'tunnels': TUNNELS['tunnels'] * 2}, WORKERS)

    def test_username_with_spaces_is_single_ssh_argument(self):
        config = {'tunnels': [dict(TUNNELS['tunnels'][0], host='SHREYAS INGLE@host')]}
        process = FakeProcess()
        with patch.object(launcher, '_available', return_value=False), \
             patch.object(launcher, '_worker_ready', return_value=True), \
             patch.object(launcher.time, 'sleep'), \
             patch.object(launcher.subprocess, 'Popen', return_value=process) as spawn, \
             patch.object(launcher.adaptive_inference, 'run', return_value={'missing': 0}):
            launcher.launch(PLAN, WORKERS, config, Path('run'))
        self.assertEqual(spawn.call_args.args[0][-1], 'SHREYAS INGLE@host')
        for host in ('-oProxyCommand=evil', 'user@-evil', 'user\nname@host', 'user @host'):
            with self.subTest(host=host), self.assertRaises(ValueError):
                launcher.validate_tunnels({'tunnels': [dict(config['tunnels'][0], host=host)]}, WORKERS)

    def test_rejects_existing_port_without_starting_ssh(self):
        with patch.object(launcher, '_available', return_value=True), \
             patch.object(launcher.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'already accepting'):
                launcher.launch(PLAN, WORKERS, TUNNELS, Path('run'))
        spawn.assert_not_called()

    def test_slow_readiness_response_obeys_absolute_deadline(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def drip():
            try:
                conn, _ = listener.accept()
                with conn:
                    conn.recv(4096)
                    payload = b'{"status":"error"}'
                    for byte in struct.pack('!I', len(payload)) + payload:
                        conn.sendall(bytes([byte]))
                        time.sleep(0.04)
            except OSError:
                pass
            finally:
                listener.close()

        thread = threading.Thread(target=drip, daemon=True)
        thread.start()
        start = time.monotonic()
        ready = launcher._worker_ready(port, start + 0.15)
        elapsed = time.monotonic() - start
        thread.join(timeout=2)
        self.assertFalse(ready)
        self.assertLess(elapsed, 0.6)


if __name__ == '__main__':
    unittest.main()
