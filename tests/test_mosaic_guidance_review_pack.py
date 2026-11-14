import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image
from exposure.mosaic_guidance_review_pack import build


class GuidanceReviewPackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        rows = []
        for color in ('red', 'blue'):
            for seed in (81380, 81381):
                for scale in (1.0, 2.0):
                    image = Image.new('RGB', (64, 64), (int(scale) * 30, seed % 255, len(color)))
                    out = io.BytesIO()
                    image.save(out, format='PNG')
                    data = out.getvalue()
                    name = f'{color}-{seed}-{scale}.png'
                    (self.source / name).write_bytes(data)
                    rows.append({'color': color, 'seed': seed, 'guidance': scale, 'file': name,
                                 'sha256': hashlib.sha256(data).hexdigest(),
                                 'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
        (self.source / 'report.json').write_text(json.dumps({'checkpoint_sha256': 'test', 'images': rows}))

    def test_pack_blinds_provenance_and_contains_complete_blank_review(self):
        result = build(self.source, self.root / 'pack')
        self.assertEqual(result['images'], 8)
        for number in range(1, 5):
            with zipfile.ZipFile(self.root / 'pack' / f'reviewer-{number}.zip') as archive:
                names = archive.namelist()
                self.assertEqual(len([n for n in names if n.startswith('images/')]), 8)
                self.assertFalse(any('private' in n or '8138' in n or 'guidance' in n for n in names))
                csv = archive.read('reviews.csv').decode().splitlines()
                self.assertEqual(len(csv), 9)
                self.assertTrue(all(line.endswith(',,') for line in csv[1:]))
                self.assertNotIn('guidance', archive.read('review.html').decode().lower())

    def test_tampered_image_is_rejected(self):
        (self.source / 'red-81380-1.0.png').write_bytes(b'not the original')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            build(self.source, self.root / 'pack')


if __name__ == '__main__':
    unittest.main()
