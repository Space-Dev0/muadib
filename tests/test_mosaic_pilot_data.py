import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from exposure.mosaic_pilot_data import CELLS, generate


class MosaicPilotDataTests(unittest.TestCase):
    def test_quotas_complexity_and_unique_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            manifest = generate(output, seed=7, train_per_cell=20, eval_per_complexity=10)
            records = manifest["records"]
            self.assertEqual(len(records), 100)
            self.assertEqual(manifest, json.loads((output / "manifest.json").read_text()))
            self.assertEqual(len({row["pixel_sha256"] for row in records}), len(records))
            for complexity, imbalance in CELLS:
                subset = [r for r in records if r["split"] == "train"
                          and r["complexity"] == complexity and r["imbalance"] == imbalance]
                self.assertEqual(Counter(r["target_color"] for r in subset),
                                 {"red": 10 if imbalance == "balanced" else 18,
                                  "blue": 10 if imbalance == "balanced" else 2})
                self.assertTrue(all(len(r["objects"]) == (2 if complexity == "low" else 3) for r in subset))
            for complexity in ("low", "high"):
                subset = [r for r in records if r["split"] == "evaluation" and r["complexity"] == complexity]
                self.assertEqual(Counter(r["target_color"] for r in subset), {"red": 5, "blue": 5})
                self.assertTrue(all(r["imbalance"] is None for r in subset))
            for row in records:
                with Image.open(output / row["image"]) as image:
                    self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(), row["pixel_sha256"])
            with self.assertRaises(FileExistsError):
                generate(output, seed=7, train_per_cell=20, eval_per_complexity=10)

    def test_invalid_quotas_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            for n, m in ((11, 10), (20, 3), (True, 10)):
                with self.assertRaises(ValueError):
                    generate(output, seed=7, train_per_cell=n, eval_per_complexity=m)
                self.assertFalse(output.exists())

    def test_seed_reproduces_scene_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = generate(Path(tmp) / "first", seed=8, train_per_cell=10, eval_per_complexity=2)
            second = generate(Path(tmp) / "second", seed=8, train_per_cell=10, eval_per_complexity=2)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
