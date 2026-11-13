import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from exposure.audit_pilot_pack import QUESTIONS, build
from exposure.audit_pilot_score import evaluate


class AuditPilotScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        source = self.base / "source"
        source.mkdir()
        checkpoint = self.base / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        for name in QUESTIONS:
            for seed in range(20, 26):
                Image.new("RGB", (64, 64), "white").save(source / f"{name}-{seed}.png")
        self.pack = self.base / "pack"
        build(source, self.pack, checkpoint, samples_per_rule=6, shared_per_rule=2, first_seed=20)
        private = self.pack / "private"
        self.reviews = {}
        shared = json.loads((private / "provenance.json").read_text())["shared_ids"]
        for i in range(1, 5):
            paths = []
            for kind, name in (("rule", "rules"), ("predicate", "predicates")):
                manifest = json.loads((private / f"reviewer-{i}-{name}.json").read_text())
                path = self.base / f"review-{i}-{kind}.csv"
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["sample_id", "judgment"])
                    for row in manifest["samples"]:
                        value = "no" if kind == "rule" and row["sample_id"] == shared[0] and i == 1 else "yes"
                        writer.writerow([row["sample_id"], value])
                paths.append(path)
            self.reviews[i] = tuple(paths)

    def tearDown(self):
        self.tmp.cleanup()

    def test_coverage_overlap_and_agreement_without_consensus(self):
        result = evaluate(self.pack, self.reviews)
        self.assertEqual(result["unique_images_reviewed"], 24)
        self.assertEqual(result["shared_images"], 8)
        self.assertEqual(result["rule_judgments"], 48)
        self.assertEqual(result["shared_rule_agreement"]["sample_count"], 8)
        self.assertEqual(result["shared_rule_agreement"]["unanimous_count"], 7)
        self.assertEqual(result["shared_predicate_agreement"]["unanimous_count"], 16)
        self.assertEqual(len(result["scores_by_reviewer"]), 4)
        self.assertNotIn("consensus", result)

    def test_rejects_missing_judgment_and_modified_frozen_prediction(self):
        target = self.reviews[1][0]
        lines = target.read_text().splitlines()
        target.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaises(ValueError):
            evaluate(self.pack, self.reviews)
        # A modified frozen prediction fails before any review is scored.
        prediction = self.pack / "private" / "reviewer-1-rule-predictions.json"
        prediction.write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluate(self.pack, self.reviews)


if __name__ == "__main__":
    unittest.main()
