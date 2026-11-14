import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional')
class WarmstartProbeTests(unittest.TestCase):
    def test_validates_update_schedule_before_opening_files(self):
        from exposure.mosaic_warmstart_probe import run
        with self.assertRaises(ValueError):
            run(Path('missing'), Path('missing'), Path('missing'), additional_updates=7,
                save_every=5, seed=1, device='cpu')

    def test_rejects_wrong_dataset_hash(self):
        from exposure.mosaic_warmstart_probe import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'bad.pt'
            torch.save({'kind': 'mosaic_multiscale_probe', 'schema_version': 1, 'step': 20000,
                        'dataset_manifest_sha256': 'different'}, checkpoint)
            with patch('exposure.mosaic_warmstart_probe.load_cells', return_value=({}, 'expected')):
                with self.assertRaisesRegex(ValueError, 'matching the dataset'):
                    run(root, checkpoint, root / 'out', additional_updates=1,
                        save_every=1, seed=1, device='cpu')
            self.assertFalse((root / 'out').exists())
