import csv
import io
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, PngImagePlugin

from exposure.audit_pilot_pack import QUESTIONS, build


class AuditPilotPackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.checkpoint = self.base / "checkpoint.pt"
        self.checkpoint.write_bytes(b"checkpoint identifier")
        for rule in QUESTIONS:
            for seed in range(20, 26):
                meta = PngImagePlugin.PngInfo()
                meta.add_text("condition", rule)
                Image.new("RGB", (64, 64), "white").save(self.source / f"{rule}-{seed}.png", pnginfo=meta)

    def tearDown(self):
        self.tmp.cleanup()

    def test_balanced_shared_assignment_and_blinding(self):
        output = self.base / "pack"
        result = build(self.source, output, self.checkpoint, seed=14,
                       samples_per_rule=6, shared_per_rule=2, first_seed=20)
        self.assertEqual(result["images"], 24)
        self.assertEqual(result["shared_images"], 8)
        self.assertEqual(result["images_per_reviewer"], [12] * 4)
        private = output / "private"
        prov = json.loads((private / "provenance.json").read_text())
        shared = set(prov["shared_ids"])
        self.assertEqual(len(prov["duplicate_image_groups"]), 1)
        seen = Counter()
        for i in range(1, 5):
            manifest = json.loads((private / f"reviewer-{i}-rules.json").read_text())["samples"]
            ids = {row["sample_id"] for row in manifest}
            self.assertTrue(shared <= ids)
            self.assertEqual(Counter(row["rule_name"] for row in manifest), {name: 3 for name in QUESTIONS})
            seen.update(ids)
            with zipfile.ZipFile(output / f"reviewer-{i}.zip") as archive:
                names = archive.namelist()
                self.assertFalse(any("prediction" in name or "manifest" in name for name in names))
                rows = list(csv.DictReader(io.StringIO(archive.read("rule-reviews.csv").decode())))
                self.assertEqual({row["sample_id"] for row in rows}, ids)
                self.assertTrue(all(row["judgment"] == "" for row in rows))
                self.assertEqual(len([name for name in names if name.startswith("images/")]), 12)
                with Image.open(io.BytesIO(archive.read(next(name for name in names if name.startswith("images/"))))) as image:
                    self.assertNotIn("condition", image.info)
                pred_rows = list(csv.DictReader(io.StringIO(archive.read("predicate-reviews.csv").decode())))
                self.assertEqual(len(pred_rows), 24)
        self.assertEqual(set(seen), set(prov["images"]))
        self.assertTrue(all(count == (4 if identifier in shared else 1) for identifier, count in seen.items()))
        with self.assertRaises(FileExistsError):
            build(self.source, output, self.checkpoint, samples_per_rule=6, shared_per_rule=2, first_seed=20)

    def test_missing_or_outside_image_does_not_create_output(self):
        path = self.source / "red-20.png"
        path.unlink()
        output = self.base / "missing"
        with self.assertRaises(FileNotFoundError):
            build(self.source, output, self.checkpoint, samples_per_rule=6, shared_per_rule=2, first_seed=20)
        self.assertFalse(output.exists())
        outside = self.base / "outside.png"
        Image.new("RGB", (64, 64), "white").save(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            build(self.source, output, self.checkpoint, samples_per_rule=6, shared_per_rule=2, first_seed=20)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
