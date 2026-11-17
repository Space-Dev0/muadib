import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exposure.checker_seg import augment, binary_metrics, build_splits, build_model, train


class SegmentationPrototypeTests(unittest.TestCase):
    def test_split_seed_and_pixels_disjoint(self):
        data = build_splits(98200, 7, 4, 4)
        seeds = [{row['seed'] for row in rows} for rows in data.values()]
        hashes = [{row['sha256'] for row in rows} for rows in data.values()]
        for sets in (seeds, hashes):
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(len(rows) for rows in data.values()), 15)
        data2 = build_splits(98200, 7, 4, 4)
        self.assertEqual([r['sha256'] for r in data['train']], [r['sha256'] for r in data2['train']])

    def test_augmentation_seeded_and_changes_input_only(self):
        row = build_splits(98210, 1, 1, 1)['train'][0]
        original_mask = row['mask'].copy()
        aug = augment(row['rgb'], 42)
        self.assertTrue(np.array_equal(aug, augment(row['rgb'], 42)))
        self.assertFalse(np.array_equal(aug, row['rgb']))
        self.assertTrue(np.array_equal(original_mask, row['mask']))
        self.assertTrue(np.array_equal(row['rgb'], augment(row['rgb'], 1, 'clean')))
        self.assertFalse(np.array_equal(row['rgb'], augment(row['rgb'], 1, 'blur_1')))
        with self.assertRaises(ValueError):
            augment(row['rgb'], 1, 'bad')

    def test_binary_metrics_and_empty_denominators(self):
        result = binary_metrics(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]))
        self.assertEqual([result[x] for x in ('tp', 'fp', 'fn', 'tn')], [1, 1, 1, 1])
        self.assertAlmostEqual(result['iou'], 1 / 3)
        self.assertIsNone(binary_metrics(np.zeros(3), np.zeros(3))['iou'])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                train(seed=0, counts=(1, 1, 1), epochs=1, batch_size=1, width=2,
                      device='cpu', output=Path(directory))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch is not installed')
    def test_model_shape_and_one_epoch_smoke(self):
        import torch
        model = build_model(4)
        self.assertEqual(model(torch.zeros(2, 3, 64, 64)).shape, (2, 1, 64, 64))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            report = train(seed=98200, counts=(3, 2, 2), epochs=1, batch_size=2,
                           width=4, device='cpu', output=out)
            self.assertEqual(len(report['split_seeds_and_pixel_sha256']['test']), 2)
            self.assertTrue((out / 'weights.pt').exists())


if __name__ == '__main__':
    unittest.main()
