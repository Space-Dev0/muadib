import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.template_balance import audit


def attr(variable, field, value):
    return {"type": "attribute", "variable": variable, "field": field, "value": value}


def rel(subject, kind, obj):
    return {"type": "relation", "subject": subject, "kind": kind, "object": obj}


class TemplateBalanceTests(unittest.TestCase):
    def test_template_level_counts_zeroes_and_variable_histogram(self):
        repeated = [attr("a", "color", "red"), attr("a", "color", "red"), rel("a", "left_of", "b")]
        other = [attr("a", "color", "blue"), attr("b", "shape", "circle"), rel("b", "above", "c")]
        report = audit({"repeated": repeated, "other": other})
        self.assertEqual(report["total_templates"], 2)
        self.assertEqual(len(report["cells"]), 1)
        cell = report["cells"][0]
        self.assertEqual((cell["k_attr"], cell["k_rel"], cell["template_count"]), (2, 1, 2))
        self.assertEqual(cell["colors"]["red"], {"count": 1, "fraction": 0.5})
        self.assertEqual(cell["colors"]["green"], {"count": 0, "fraction": 0.0})
        self.assertEqual(cell["shapes"]["square"], {"count": 0, "fraction": 0.0})
        self.assertEqual(cell["relations"]["left_of"], {"count": 1, "fraction": 0.5})
        self.assertEqual(cell["object_variable_histogram"], {"2": 1, "3": 1})
        self.assertEqual(len(report["missing_expected_cells"]), 8)
        self.assertIn("not a balance test", report["scope"])

    def test_audit_deterministic_and_multiple_cells(self):
        rules = {"z": [attr("a", "shape", "square")],
                 "a": [attr("x", "color", "blue"), rel("x", "above", "y")]}
        first = audit(rules)
        self.assertEqual(first, audit(rules))
        self.assertEqual([(c["k_attr"], c["k_rel"]) for c in first["cells"]], [(1, 0), (1, 1)])
        self.assertEqual(first["cells"][0]["relations"]["above"]["count"], 0)

    def test_cli_rejects_duplicate_keys_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            rules = Path(directory) / "rules.json"
            output = Path(directory) / "report.json"
            rules.write_text('{"x": [], "x": []}', encoding="utf-8")
            run = subprocess.run([sys.executable, "-m", "exposure.template_balance", "--rules", str(rules),
                                  "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("Duplicate JSON key", run.stderr)
            self.assertFalse(output.exists())

    def test_cli_emits_reproducible_json(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rules = directory / "rules.json"
            output = directory / "report.json"
            rules.write_text(json.dumps({"red": [attr("a", "color", "red")]}), encoding="utf-8")
            cmd = [sys.executable, "-m", "exposure.template_balance", "--rules", str(rules), "--output", str(output)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            first = output.read_text(encoding="utf-8")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertEqual(first, output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
