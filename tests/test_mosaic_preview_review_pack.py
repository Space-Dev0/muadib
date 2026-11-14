import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image

from exposure.mosaic_preview_review_pack import CELLS, build


class MosaicReviewPackTests(unittest.TestCase):
    def test_blinding_and_complete_fixed_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'pack'
            for complexity, imbalance in CELLS:
                cell = source / f'{complexity}-{imbalance}'
                cell.mkdir(parents=True)
                report = {'cell': [complexity, imbalance], 'training_seed': 81322, 'width': 32,
                          'batch_size': 4, 'learning_rate': 0.0001, 'device': 'cuda', 'torch_version': 'test',
                          'dataset_manifest_sha256': 'manifest', 'source_sha256': 'source',
                          'sample_seeds': [81330, 81331], 'sampler_steps': 50,
                          'snapshots': [{'step': 20000, 'checkpoint_sha256': 'checkpoint'}]}
                (cell / 'report.json').write_text(json.dumps(report))
                for label in ('red', 'blue'):
                    for seed in (81330, 81331):
                        Image.new('RGB', (64, 64), 'white').save(cell / f'20000-{label}-{seed}.png')
            result = build(source, output)
            self.assertEqual(result['images'], 16)
            self.assertEqual(len(json.loads((output / 'private' / 'predictions.json').read_text())), 16)
            with zipfile.ZipFile(output / 'reviewer-1.zip') as z:
                self.assertEqual(len([n for n in z.namelist() if n.endswith('.png')]), 16)
                self.assertFalse(any('prediction' in n or 'private' in n for n in z.namelist()))
                page = z.read('review.html').decode()
                self.assertNotIn('low-balanced', page)
                self.assertIn('exactly two distinct objects', page)
                self.assertIn('exactly three distinct objects', page)
            with self.assertRaises(FileExistsError):
                build(source, output)

    def test_rejects_overlapping_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(ValueError):
                build(path, path / 'pack')
