import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from exposure.mosaic_matched_eval_pack import CELLS, SEEDS, build
from exposure.mosaic_matched_eval_score import score


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class MatchedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.source = self.root / 'source'; self.source.mkdir()
        (self.source / 'COMPLETE').write_text('done')
        protocol = {'seeds': SEEDS, 'guidance': 1.0,
                    'cells': {cell: {'sha256': cell} for cell in CELLS}}
        (self.source / 'protocol.json').write_text(json.dumps(protocol))
        for cell in CELLS:
            directory = self.source / cell; directory.mkdir(); rows = []
            for color in ('red', 'blue'):
                for seed in SEEDS:
                    image = Image.new('RGB', (64, 64), (255, 255, 255))
                    buf = io.BytesIO(); image.save(buf, format='PNG'); raw = buf.getvalue()
                    name = f'{color}-{seed}.png'; (directory / name).write_bytes(raw)
                    rows.append({'color': color, 'seed': seed, 'guidance': 1.0, 'file': name,
                                 'sha256': sha(raw), 'pixel_sha256': sha(image.tobytes())})
            report = {'checkpoint_sha256': cell, 'seeds': SEEDS, 'guidance': [1.0], 'images': rows}
            (directory / 'report.json').write_text(json.dumps(report))

    def test_pack_and_unresolved_review_bounds(self):
        pack = self.root / 'pack'; self.assertEqual(build(self.source, pack)['images'], 64)
        paths = []
        for number in range(4):
            with zipfile.ZipFile(pack / f'reviewer-{number + 1}.zip') as archive:
                rows = list(csv.DictReader(io.StringIO(archive.read('reviews.csv').decode())))
                self.assertEqual(len([x for x in archive.namelist() if x.startswith('images/')]), 64)
                self.assertFalse(any('private' in x for x in archive.namelist()))
            for row in rows:
                row.update(target_circle='yes', exactly_two_objects='yes', exactly_three_objects='no')
            if number == 0:
                rows[0]['target_circle'] = 'unsure'
            path = self.root / f'review-{number}.csv'; paths.append(path)
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        result = score(pack, paths)
        self.assertEqual(sum(c[color]['unresolved'] for c in result['cells_by_color'].values()
                             for color in ('red', 'blue')), 1)
        self.assertEqual(sum(c[color]['yes'] for c in result['cells_by_color'].values()
                             for color in ('red', 'blue')), 63)
        paths[0].write_text(paths[0].read_text().replace('matched-0001', 'matched-0002'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            score(pack, paths)

    def test_hash_tampering_is_rejected(self):
        (self.source / CELLS[0] / f'red-{SEEDS[0]}.png').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            build(self.source, self.root / 'pack')


if __name__ == '__main__':
    unittest.main()
