import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.measure import estimate, load_rules, wilson_interval
from exposure.rules import Attribute


class ExposureMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(Path("examples/exposure_rules.json").read_text())
        self.rules = load_rules(self.data)

    def test_matrix_direction_implications_and_reproducibility(self):
        result = estimate(self.rules, samples=100, seed=42)
        self.assertEqual(result, estimate(self.rules, samples=100, seed=42))
        columns = {column["condition"]: column for column in result["columns"]}
        for name, column in columns.items():
            self.assertEqual(column["measurements"][name]["successes"], 100)
            self.assertGreaterEqual(column["total_attempts"], 100)
        complex_column = columns["red_left_of_square"]["measurements"]
        self.assertEqual(complex_column["red"]["successes"], 100)
        self.assertEqual(complex_column["square"]["successes"], 100)
        self.assertLess(columns["red"]["measurements"]["red_left_of_square"]["successes"], 100)
        self.assertGreater(columns["blue_above_circle"]["measurements"]["red"]["successes"], 0)

    def test_wilson_bounds_at_extremes(self):
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertGreater(wilson_interval(0, 10)[1], 0)
        self.assertLess(wilson_interval(10, 10)[0], 1)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)

    def test_invalid_and_impossible_inputs_fail(self):
        with self.assertRaises(ValueError):
            load_rules({})
        with self.assertRaises(ValueError):
            estimate(self.rules, samples=0, seed=1)
        impossible = {"impossible": (Attribute("a", "color", "red"), Attribute("a", "color", "blue"))}
        with self.assertRaisesRegex(ValueError, "Condition 'impossible', sample 1"):
            estimate(impossible, samples=1, seed=1, max_attempts=5)

    def test_cli_preserves_rule_definitions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            subprocess.run([sys.executable, "-m", "exposure.measure", "--rules", "examples/exposure_rules.json",
                            "--samples", "10", "--seed", "4", "--output", str(output)],
                           check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["rules"], self.data)
            self.assertEqual(result["samples_per_condition"], 10)

    def test_cli_rejects_duplicate_rule_names(self):
        with tempfile.TemporaryDirectory() as directory:
            rules_path = Path(directory) / "rules.json"
            output = Path(directory) / "report.json"
            rules_path.write_text('{"red": [], "red": []}')
            result = subprocess.run([sys.executable, "-m", "exposure.measure", "--rules", str(rules_path),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate JSON key", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
