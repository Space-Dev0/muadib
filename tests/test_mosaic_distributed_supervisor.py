import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class DistributedSupervisorTests(unittest.TestCase):
    def test_latest_recovers_completed_and_validates_orphan_checkpoints(self):
        from exposure import mosaic_distributed_supervisor as sup
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / 'workspace'
            original = workspace / 'outputs' / 'mosaic-matched-81570-low-balanced-continued'
            original.mkdir(parents=True)
            (original / 'warmstart-150000.pt').write_bytes(b'old')
            (original / 'report.json').write_text(json.dumps({'snapshots': [{'step': 150000,
                    'checkpoint_sha256': sup.digest(original / 'warmstart-150000.pt')}]}))
            output = workspace / 'outputs' / 'mosaic-distributed-81570'
            orphan = output / 'low-balanced' / 'segment-160000-interrupted'
            orphan.mkdir(parents=True)
            (orphan / 'warmstart-160000.pt').write_bytes(b'new')
            with mock.patch.object(sup, 'checkpoint') as validate:
                step, path, sha, mode = sup.latest(workspace, output, 'low-balanced')
            self.assertEqual(step, 160000)
            self.assertEqual(path, (orphan / 'warmstart-160000.pt').resolve())
            self.assertEqual(sha, sup.digest(path))
            self.assertEqual(mode, 'resume')
            self.assertEqual(validate.call_count, 2)

    def test_missing_sources_requires_bootstrap(self):
        from exposure import mosaic_distributed_supervisor as sup
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(sup.latest(root, root / 'output', 'high-imbalanced'))

    def test_run_namespace_does_not_use_other_block_checkpoints(self):
        from exposure import mosaic_distributed_supervisor as sup
        previous = sup.RUN_ID
        self.addCleanup(setattr, sup, 'RUN_ID', previous)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); workspace = root / 'workspace'
            old = workspace / 'outputs' / 'mosaic-matched-81570-low-balanced-continued'
            old.mkdir(parents=True)
            (old / 'report.json').write_text('{}')
            sup.RUN_ID = 82600
            self.assertIsNone(sup.latest(workspace, root / 'new-output', 'low-balanced'))


if __name__ == '__main__':
    unittest.main()
