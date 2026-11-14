"""Tests for the paired class-conditioning probe."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from PIL import Image
    from exposure.mosaic_pilot_data import CELLS
    from exposure.mosaic_toy_condition_probe import compare
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class ConditionProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        for complexity, imbalance in CELLS:
            checkpoint = {"schema_version": 1, "kind": "mosaic_toy_velocity", "cell": [complexity, imbalance],
                          "seed": 1, "steps": 2, "batch_size": 2, "width": 8, "learning_rate": 0.0001,
                          "device": "cpu", "torch_version": "x", "python_version": "x",
                          "manifest_sha256": "m", "trainer_sha256": "t", "initial_parameter_sha256": "i"}
            torch.save(checkpoint, self.path / f"{complexity}-{imbalance}.pt")

    def test_paired_seeds_and_computation(self):
        seen = []
        def fake_sample(checkpoint, *, label, seed, steps, device):
            seen.append((tuple(checkpoint['cell']), label, seed))
            return Image.new("RGB", (64, 64), (255, 0, 0) if label == 'red' else (0, 0, 255))
        with patch('exposure.mosaic_toy_condition_probe.sample', side_effect=fake_sample):
            result = compare(self.path, first_seed=12, seeds=2, steps=3, device='cpu')
        self.assertEqual(len(seen), 16)
        self.assertEqual({x[2] for x in seen}, {12, 13})
        for cell in result['results']:
            self.assertAlmostEqual(cell['mean_absolute_rgb_difference'], 2/3)
            self.assertEqual(cell['mean_paired_red_minus_blue_contrast'], 2.0)
        for complexity, imbalance in CELLS:
            for seed in (12, 13):
                self.assertEqual({label for c, label, s in seen if c == (complexity, imbalance) and s == seed}, {'red', 'blue'})

    def test_mismatched_training_configuration_rejected(self):
        path = self.path / 'high-imbalanced.pt'
        cp = torch.load(path, weights_only=True)
        cp['seed'] = 99
        torch.save(cp, path)
        with patch('exposure.mosaic_toy_condition_probe.sample', return_value=Image.new('RGB', (64, 64))):
            with self.assertRaisesRegex(ValueError, 'matched'):
                compare(self.path, first_seed=0, seeds=1, steps=2, device='cpu')


if __name__ == '__main__':
    unittest.main()
