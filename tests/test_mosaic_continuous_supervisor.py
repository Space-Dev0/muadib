import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional')
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        from exposure import mosaic_continuous_supervisor as supervisor
        self.s = supervisor
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'warmstart-030000.pt'
        self.checkpoint.write_bytes(b'checkpoint')
        names = ('pilot_train', 'mosaic_pilotunet_probe', 'mosaic_pilotunet_warmstart',
                 'mosaic_toy_train', 'velocity_probe', 'identity_eps_finetune', 'measure', 'render', 'rules')
        hashes = {name: supervisor.digest(Path(importlib.import_module('exposure.' + name).__file__))
                  for name in names}
        self.cp = {'cell': ['low', 'balanced'],
                   'config': {'width': 16, 'timesteps': 1000, 'batch_size': 4, 'seed': 81377,
                              'lr': 0.0001, 'dropout': 0.1, 'ema_decay': 0.999, 'deterministic': False},
                   'report': {'cumulative_step': 30000}, 'rng': {'device': 'cuda', 'torch_cuda': [b'x']},
                   'provenance': {'torch': str(torch.__version__), 'python': sys.version,
                                  'dataset': {'manifest_sha256': supervisor.MANIFEST},
                                  'runtime': {'code_sha256': hashes}}}

    def test_checkpoint_validates_hash_config_and_runtime(self):
        with patch.object(self.s, 'load_warmstart_checkpoint', return_value=self.cp), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=1):
            sha = self.s.validate_checkpoint(self.checkpoint, 30000)
            self.assertEqual(sha, self.s.digest(self.checkpoint))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.s.validate_checkpoint(self.checkpoint, 30000, 'wrong')
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                self.s.validate_checkpoint(self.checkpoint, 40000)
            self.cp['rng']['torch_cuda'] = []
            with self.assertRaisesRegex(ValueError, 'runtime'):
                self.s.validate_checkpoint(self.checkpoint, 30000)

    def test_completed_report_requires_matching_snapshot_hash(self):
        directory = self.root / 'segment'
        directory.mkdir()
        (directory / 'report.json').write_text(json.dumps({'snapshots': [{'step': 40000, 'checkpoint_sha256': 'abc'}]}))
        with patch.object(self.s, 'validate_checkpoint', return_value='abc') as check:
            path, sha = self.s.completed_report_checkpoint(directory, 40000)
        self.assertEqual(path.name, 'warmstart-040000.pt')
        self.assertEqual(sha, 'abc')
        check.assert_called_once_with(directory / 'warmstart-040000.pt', 40000, 'abc')
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.s.completed_report_checkpoint(directory, 50000)

    def test_stop_marker_prevents_predecessor_and_child_launch(self):
        workspace = self.root / 'workspace'
        workspace.mkdir()
        control = self.root / 'control'
        control.mkdir()
        (control / 'STOP').write_text('stop')
        with patch.object(self.s, 'predecessor_running') as predecessor, \
             patch.object(self.s.subprocess, 'Popen') as launch:
            self.s.supervise(workspace, control)
        predecessor.assert_not_called()
        launch.assert_not_called()


    def test_completed_segment_is_recorded_before_stop(self):
        workspace = self.root / 'workspace'
        workspace.mkdir()
        control = self.root / 'control'
        source = workspace / 'source.pt'
        source.write_bytes(b'source')
        def complete(directory, step):
            (control / 'STOP').write_text('stop')
            return Path(directory) / 'warmstart-040000.pt', 'new-hash'
        with patch.object(self.s, 'predecessor_running', return_value=False), \
             patch.object(self.s, 'initial_checkpoint', return_value=(30000, source, 'old-hash', 'known')), \
             patch.object(self.s.subprocess, 'Popen') as launch, \
             patch.object(self.s, 'completed_report_checkpoint', side_effect=complete):
            launch.return_value.poll.return_value = 0
            launch.return_value.returncode = 0
            self.s.supervise(workspace, control)
        state = json.loads((control / 'state.json').read_text())
        self.assertEqual(state['step'], 40000)
        self.assertEqual(state['sha256'], 'new-hash')
        launch.assert_called_once()


    def test_stop_during_segment_terminates_owned_windows_process_tree(self):
        workspace = self.root / 'workspace'
        workspace.mkdir()
        control = self.root / 'control'
        source = workspace / 'source.pt'
        source.write_bytes(b'source')
        def launch(*args, **kwargs):
            (control / 'STOP').write_text('stop')
            process = unittest.mock.MagicMock()
            process.pid = 4567
            process.poll.return_value = None
            return process
        with patch.object(self.s, 'predecessor_running', return_value=False), \
             patch.object(self.s, 'initial_checkpoint', return_value=(30000, source, 'hash', 'known')), \
             patch.object(self.s.subprocess, 'Popen', side_effect=launch), \
             patch.object(self.s.subprocess, 'run') as taskkill:
            taskkill.return_value.returncode = 0
            self.s.supervise(workspace, control)
        self.assertEqual(taskkill.call_args.args[0], ['taskkill.exe', '/PID', '4567', '/T', '/F'])


if __name__ == '__main__':
    unittest.main()
