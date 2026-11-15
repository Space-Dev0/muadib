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
class MatchedCellTests(unittest.TestCase):
    def test_loads_selected_cell_and_records_wrapper_fingerprint(self):
        from exposure import mosaic_matched_cell as wrapper
        from exposure import mosaic_pilotunet_probe as probe
        original = (probe.CELL, probe.load_cell, probe._runtime)
        self.addCleanup(lambda: self.restore(probe, original))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{'split': 'train', 'complexity': 'high', 'imbalance': 'imbalanced',
                        'target_color': color, 'image': f'{color}.png', 'pixel_sha256': color}
                       for color in ('red', 'blue')]
            (root / 'manifest.json').write_text(json.dumps({'records': records}))
            entries = [(torch.zeros((3, 2, 2)), 0), (torch.zeros((3, 2, 2)), 1)]
            with mock.patch.object(wrapper, 'load_cells', return_value=({('high', 'imbalanced'): entries}, 'manifest')):
                wrapper.configure(('high', 'imbalanced'))
                images, colors, manifest_hash, selected, data_hash = probe.load_cell(root)
            self.assertEqual(probe.CELL, ('high', 'imbalanced'))
            self.assertEqual(colors, ['red', 'blue'])
            self.assertEqual(len(images), 2)
            self.assertEqual(manifest_hash, 'manifest')
            self.assertEqual([r['image'] for r in selected], ['red.png', 'blue.png'])
            self.assertEqual(len(data_hash), 64)
            self.assertIn('mosaic_matched_cell', probe._runtime('cpu')['code_sha256'])
            with self.assertRaises(ValueError):
                wrapper.configure(('unknown', 'balanced'))

    @staticmethod
    def restore(probe, original):
        probe.CELL, probe.load_cell, probe._runtime = original


if __name__ == '__main__':
    unittest.main()
