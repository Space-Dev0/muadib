import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exposure.measure import estimate, load_rules, wilson_interval
from exposure.rules import Attribute, SceneObject


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
            self.assertEqual(column["measurements"][name]["wilson_95"], [1.0, 1.0])
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

    def test_subsets_keep_shared_and_distinct_variables(self):
        result = estimate(self.rules, samples=80, seed=7, include_subsets=True)
        column = next(c for c in result["columns"] if c["condition"] == "red_left_of_square")
        subsets = column["predicate_subsets"]["red_left_of_square"]
        self.assertEqual([entry["predicate_indices"] for entry in subsets],
                         [[0], [1], [2], [0, 1], [0, 2], [1, 2]])
        self.assertTrue(all(entry["successes"] == 80 for entry in subsets))
        self.assertTrue(all(entry["wilson_95"] == [1.0, 1.0] for entry in subsets))
        red_column = next(c for c in result["columns"] if c["condition"] == "red")
        self.assertEqual(red_column["predicate_subsets"]["red_left_of_square"][0]["successes"], 80)
        self.assertLess(red_column["predicate_subsets"]["red_left_of_square"][4]["successes"], 80)
        self.assertNotIn("predicate_subsets", estimate(self.rules, samples=1, seed=7)["columns"][0])

    def test_pair_requires_distinct_objects_when_variables_differ(self):
        rules = {"red": (Attribute("a", "color", "red"),),
                 "red_and_square": (Attribute("a", "color", "red"),
                                    Attribute("b", "shape", "square"))}
        objects = (SceneObject("red", "square", 1, 1),
                   SceneObject("blue", "circle", 2, 2),
                   SceneObject("green", "triangle", 3, 3))
        # Isolate the red column with a known scene whose only red object is also
        # the only square. Singleton events hold, but their conjunction does not.
        pair_scene = (SceneObject("red", "circle", 1, 1),
                      SceneObject("blue", "square", 2, 2),
                      SceneObject("green", "triangle", 3, 3))
        def sample(_rng, predicates, _max_attempts):
            return (objects if predicates == rules["red"] else pair_scene), 1

        with patch("exposure.measure.sample_conditioned", side_effect=sample):
            column = estimate(rules, samples=1, seed=1, include_subsets=True)["columns"][0]
        entries = column["predicate_subsets"]["red_and_square"]
        self.assertEqual([entry["successes"] for entry in entries], [1, 1, 0])

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
                            "--samples", "10", "--seed", "4", "--include-subsets", "--output", str(output)],
                           check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["rules"], self.data)
            self.assertEqual(result["samples_per_condition"], 10)
            self.assertIn("predicate_subsets", result["columns"][0])

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
