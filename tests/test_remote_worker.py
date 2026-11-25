import base64
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ops import inference_launcher, remote_worker


class RemoteWorkerTests(unittest.TestCase):
    def _fake_reader(self, output: bytes):
        worker = remote_worker.OwnedWorker.__new__(remote_worker.OwnedWorker)
        worker.process = MagicMock()
        worker.process.stdout = io.BytesIO(output)
        worker._condition = threading.Condition()
        worker._ready = worker._stop_seen = worker._reader_finished = False
        worker._diagnostic = None
        worker.expected = 'MUADIB_READY_private-token_digest'
        worker.stop_prefix = 'MUADIB_STOP_private-token_'
        return worker

    def test_reader_start_failure_reaps_owned_child_and_preserves_original_error(self):
        worker_config = {'platform': 'linux', 'python': '/usr/bin/python3', 'cwd': '/tmp',
                         'checkpoint': '/tmp/model.pt', 'format': 'warmstart', 'device': 'cpu',
                         'lease_seconds': 60}
        tunnel = {'start_worker': worker_config, 'ssh_port': 22, 'remote_port': 5000,
                  'host': 'user@host'}
        worker = {'checkpoint_sha256': 'a' * 64}
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(remote_worker.subprocess, 'Popen', return_value=process), \
             patch.object(remote_worker.threading.Thread, 'start', side_effect=RuntimeError('thread limit')), \
             self.assertWarnsRegex(RuntimeWarning, 'cleanup unconfirmed'):
            with self.assertRaisesRegex(RuntimeError, 'thread limit'):
                remote_worker.OwnedWorker(tunnel, worker)
        process.stdin.close.assert_called_once()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=3)
        process.stdout.close.assert_called_once()

    def test_reader_closes_its_stdout_after_eof(self):
        process = subprocess.Popen([sys.executable, '-c', 'print("hello")'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        worker = remote_worker.OwnedWorker.__new__(remote_worker.OwnedWorker)
        worker.process = process
        worker._condition = threading.Condition()
        worker._ready = worker._stop_seen = worker._reader_finished = False
        worker._diagnostic = None
        worker.expected = 'READY'
        worker.stop_prefix = 'STOP'
        try:
            worker._read_lines()
            self.assertTrue(process.stdout.closed)
            self.assertTrue(worker._reader_finished)
            self.assertEqual(process.wait(timeout=3), 0)
        finally:
            process.stdin.close()
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_startup_error_classification_redacts_arbitrary_remote_output(self):
        worker = self._fake_reader(b'private-token private-path encoded-command\n'
                                   b'checkpoint SHA-256 mismatch; private-token private-path\n')
        worker._read_lines()
        with self.assertRaises(RuntimeError) as error:
            worker.wait_ready(1)
        self.assertIn('checkpoint SHA-256 mismatch', str(error.exception))
        self.assertNotIn('private-token', str(error.exception))
        self.assertNotIn('private-path', str(error.exception))
        self.assertNotIn('encoded-command', str(error.exception))

    def test_unknown_startup_error_does_not_expose_remote_output(self):
        worker = self._fake_reader(b'unexpected secret output\n')
        worker._read_lines()
        with self.assertRaisesRegex(RuntimeError, '^remote owned worker exited before readiness$'):
            worker.wait_ready(1)

    def test_control_markers_survive_noisy_output_without_retaining_logs(self):
        worker = self._fake_reader(b'noise\n' * 1000 + b'MUADIB_READY_private-token_digest\n'
                                   + b'noise\n' * 1000 + b'MUADIB_STOP_private-token_EOF\n')
        worker._read_lines()
        worker.wait_ready(1)
        self.assertTrue(worker._stop_seen)
        self.assertTrue(worker._reader_finished)
        self.assertIsNone(worker._diagnostic)

    def test_close_joins_reader_and_confirms_marker_before_closing_stdout(self):
        script = 'import sys; print("READY", flush=True); sys.stdin.buffer.read(); print("STOP_EOF", flush=True)'
        process = subprocess.Popen([sys.executable, '-u', '-c', script], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, bufsize=0)
        worker = remote_worker.OwnedWorker.__new__(remote_worker.OwnedWorker)
        worker.process = process
        worker._condition = threading.Condition()
        worker._ready = worker._stop_seen = worker._reader_finished = False
        worker._diagnostic = None
        worker.cleanup_confirmed = False
        worker.expected = 'READY'
        worker.stop_prefix = 'STOP_'
        worker.reader = threading.Thread(target=worker._read_lines, daemon=True)
        worker.reader.start()
        try:
            worker.wait_ready(3)
            worker.close()
            self.assertTrue(worker.cleanup_confirmed)
            self.assertTrue(process.stdout.closed)
            self.assertFalse(worker.reader.is_alive())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def _workspace(self, root):
        package = Path(root) / 'ops'
        package.mkdir()
        (package / '__init__.py').write_text('')
        (package / 'distributed_inference.py').write_text('''\
import argparse, socket, time
p=argparse.ArgumentParser(); p.add_argument('worker'); p.add_argument('--checkpoint'); p.add_argument('--format'); p.add_argument('--device'); p.add_argument('--port', type=int)
a=p.parse_args()
s=socket.socket(); s.bind(('127.0.0.1', a.port)); s.listen()
while True: time.sleep(1)
''')
        checkpoint = Path(root) / 'model checkpoint.pt'
        checkpoint.write_bytes(b'test checkpoint')
        return checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def _start_local(self, root, checkpoint, digest, port, lease=30):
        payload = {'cwd': root, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                   'port': port, 'token': 'testtoken', 'format': 'warmstart', 'device': 'cpu',
                   'lease_seconds': lease}
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        program = 'PAYLOAD=' + repr(encoded) + '\n' + remote_worker._PROGRAM
        return subprocess.Popen([sys.executable, '-u', '-c', program], cwd=root,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def test_self_exit_on_stdin_eof_and_port_released(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint, digest = self._workspace(root)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            process = self._start_local(root, checkpoint, digest, port)
            try:
                self.assertEqual(process.stdout.readline().decode().strip(), f'MUADIB_READY_testtoken_{digest}')
                process.stdin.close()
                self.assertEqual(process.stdout.readline().decode().strip(), 'MUADIB_STOP_testtoken_EOF')
                self.assertEqual(process.wait(timeout=5), 0)
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', port))
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
                process.stdout.close()

    def test_existing_listener_survives_failed_owned_worker_start(self):
        with tempfile.TemporaryDirectory() as root, socket.socket() as existing:
            checkpoint, digest = self._workspace(root)
            existing.bind(('127.0.0.1', 0)); existing.listen(); port = existing.getsockname()[1]
            process = self._start_local(root, checkpoint, digest, port)
            try:
                output = process.stdout.read().decode()
                self.assertNotEqual(process.wait(timeout=5), 0)
                self.assertNotIn('MUADIB_READY_', output)
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    pass
            finally:
                process.stdin.close(); process.stdout.close()

    def test_wrong_checkpoint_refuses_start(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint, _ = self._workspace(root)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            process = self._start_local(root, checkpoint, 'a' * 64, port)
            try:
                output = process.stdout.read().decode()
                self.assertNotEqual(process.wait(timeout=5), 0)
                self.assertIn('checkpoint SHA-256 mismatch', output)
                self.assertNotIn('MUADIB_READY_', output)
            finally:
                process.stdin.close(); process.stdout.close()

    def test_lease_expires_without_stdin_eof(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint, digest = self._workspace(root)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
            process = self._start_local(root, checkpoint, digest, port, lease=.3)
            try:
                self.assertIn('MUADIB_READY_testtoken_', process.stdout.readline().decode())
                self.assertEqual(process.stdout.readline().decode().strip(), 'MUADIB_STOP_testtoken_LEASE')
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                process.stdin.close(); process.stdout.close()
                if process.poll() is None:
                    process.kill(); process.wait()

    def test_windows_paths_are_encoded_and_shell_metacharacters_rejected(self):
        config = {'platform': 'windows', 'python': r'C:\\Users\\Name With Space\\python.exe',
                  'cwd': r'C:\\Users\\Name With Space\\project', 'checkpoint': 'x & y.pt',
                  'format': 'warmstart', 'device': 'cuda', 'lease_seconds': 120}
        remote_worker.validate_worker_config(config)
        command = remote_worker._command(config, 'a' * 64, 1234, 'f' * 32)
        self.assertTrue(command.startswith('"C:'))
        self.assertNotIn('x & y.pt', command)
        with self.assertRaises(ValueError):
            remote_worker.validate_worker_config(dict(config, python='python & calc'))

    def test_optional_worker_cleanup_on_coordinator_error(self):
        config = {'tunnels': [{'worker_id': 'node', 'host': 'user@host', 'ssh_port': 22, 'remote_port': 50300,
                               'start_worker': {'platform': 'linux', 'python': '/usr/bin/python3', 'cwd': '/tmp',
                                                'checkpoint': '/tmp/model.pt', 'format': 'warmstart',
                                                'device': 'cpu', 'lease_seconds': 60}}]}
        worker = {'id': 'node', 'port': 50400, 'checkpoint_sha256': 'a' * 64}
        plan = {'schema_version': 1, 'jobs': [{'id': 'one', 'stratum': 's', 'checkpoint_sha256': 'a' * 64,
                                               'color': 'red', 'seed': 1}]}
        with patch.object(inference_launcher, '_available', return_value=False), \
             patch.object(inference_launcher.remote_worker, 'OwnedWorker') as owner, \
             patch.object(inference_launcher.subprocess, 'Popen') as tunnel, \
             patch.object(inference_launcher, '_worker_ready', return_value=True), \
             patch.object(inference_launcher.time, 'sleep'), \
             patch.object(inference_launcher.adaptive_inference, 'run', side_effect=RuntimeError('fail')), \
             self.assertWarnsRegex(RuntimeWarning, 'lease is shorter'):
            tunnel.return_value.poll.return_value = None
            with self.assertRaisesRegex(RuntimeError, 'fail'):
                inference_launcher.launch(plan, [worker], config, Path('/tmp/unused'))
            owner.return_value.wait_ready.assert_called_once()
            owner.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
