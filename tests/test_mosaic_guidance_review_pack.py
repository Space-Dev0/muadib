import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image
from exposure.mosaic_guidance_review_pack import build
from exposure.mosaic_guidance_review_score import score


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

    def test_fresh_pack_requires_exact_samples_and_checkpoint(self):
        rows = []
        for color in ('red', 'blue'):
            for seed in range(81530, 81546):
                image = Image.new('RGB', (64, 64), (seed % 255, len(color), 10))
                out = io.BytesIO()
                image.save(out, format='PNG')
                raw = out.getvalue()
                name = f'{color}-{seed}.png'
                (self.source / name).write_bytes(raw)
                rows.append({'color': color, 'seed': seed, 'guidance': 1.0, 'file': name,
                             'sha256': hashlib.sha256(raw).hexdigest(),
                             'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
        report = {'checkpoint_sha256': 'wrong', 'images': rows}
        (self.source / 'report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            build(self.source, self.root / 'bad', fresh_81530=True)
        report['checkpoint_sha256'] = '84dac304a3a2b67ca8342f331b05adea940da746293dd7a3002bb902c43d91d4'
        (self.source / 'report.json').write_text(json.dumps(report))
        self.assertEqual(build(self.source, self.root / 'fresh', fresh_81530=True)['images'], 32)
        with zipfile.ZipFile(self.root / 'fresh' / 'reviewer-1.zip') as archive:
            self.assertEqual(len(archive.read('reviews.csv').decode().splitlines()), 33)

    def test_score_preserves_disagreement_and_rejects_incomplete_reviews(self):
        pack = self.root / 'pack'
        build(self.source, pack)
        files = []
        for number in range(4):
            path = self.root / f'review-{number}.csv'
            with zipfile.ZipFile(pack / f'reviewer-{number + 1}.zip') as archive:
                text = archive.read('reviews.csv').decode().replace(',,', ',yes,yes')
            if number == 0:
                text = text.replace('readiness-001,yes,yes', 'readiness-001,no,unsure')
            path.write_text(text)
            files.append(path)
        result = score(pack, files)
        self.assertEqual(result['questions']['target_circle']['yes_no_disagreement'], 1)
        self.assertEqual(result['questions']['exactly_two_objects']['any_unsure'], 1)
        self.assertEqual(result['questions']['target_circle']['unanimous_yes'], 7)
        files[0].write_text(files[0].read_text().replace('readiness-001', 'readiness-002'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            score(pack, files)


if __name__ == '__main__':
    unittest.main()
