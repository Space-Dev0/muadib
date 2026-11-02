import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.audit_agreement import summarize


def sample(identifier):
    return {"sample_id": identifier, "image_path": identifier + ".png", "rule_name": "red",
            "predicates": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}],
            "k_attr": 1, "k_rel": 0}


class AuditAgreementTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"samples": [sample("one"), sample("two"), sample("three")]}
        self.reviews = {
            "a": {"one": True, "two": False, "three": True},
            "b": {"one": True, "two": True, "three": False},
            "c": {"one": True, "two": False, "three": False},
        }

    def test_counts_and_unresolved_disagreements(self):
        report = summarize(self.manifest, self.reviews)
        self.assertEqual(report["unanimous_count"], 1)
        self.assertEqual([x["sample_id"] for x in report["disagreements"]], ["three", "two"])
        pair = report["pairs"][0]
        self.assertEqual((pair["reviewer_a"], pair["reviewer_b"]), ("a", "b"))
        self.assertEqual(pair["counts"], {"yes_yes": 1, "yes_no": 1, "no_yes": 1, "no_no": 0})
        self.assertAlmostEqual(pair["agreement_fraction"], 1 / 3)

    def test_requires_complete_independent_review_files(self):
        with self.assertRaises(ValueError):
            summarize(self.manifest, {"a": self.reviews["a"]})
        with self.assertRaises(ValueError):
            summarize(self.manifest, {"a": self.reviews["a"], "b": {"one": True}})
        with self.assertRaises(ValueError):
            summarize(self.manifest, {"a": self.reviews["a"], "b": {"one": 1, "two": False, "three": True}})

    def test_cli_rejects_output_alias_of_a_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
            first, second = root / "first.csv", root / "second.csv"
            content = "sample_id,judgment\none,yes\ntwo,no\nthree,yes\n"
            first.write_text(content, encoding="utf-8")
            second.write_text(content, encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "exposure.audit_agreement", "--manifest", str(manifest),
                                     "--review", str(first), "--review", str(second), "--output", str(first)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(first.read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
