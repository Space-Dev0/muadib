import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.design_bounds import _lift_bounds, feasible_lift_bounds
from exposure.measure import estimate, load_rules
from exposure.mixture import summarize


class DesignBoundsTests(unittest.TestCase):
    def test_zero_lower_exposure_retains_finite_upper_bound(self):
        lower, upper, status = _lift_bounds(0.0, 0.5, 0.25, [0.5, 0.5])
        self.assertIsNone(lower)
        self.assertAlmostEqual(upper, math.log(2))
        self.assertEqual(status, "zero_lower_exposure")

    @classmethod
    def setUpClass(cls):
        cls.rule_data = json.loads(Path("examples/exposure_rules.json").read_text())
        cls.report = estimate(load_rules(cls.rule_data), samples=40, seed=42,
                              include_subsets=True)
        cls.report["rules"] = cls.rule_data
        cls.weights = {name: 0.25 for name in cls.rule_data}

    def test_bounds_contain_reference_lift_and_are_ordered(self):
        result = feasible_lift_bounds(self.report, self.weights)
        self.assertEqual(result["summary"]["rule_count"], len(self.rule_data))
        reference = summarize(self.report, self.weights)["results"]
        for entry in result["rules"]:
            self.assertEqual(entry.get("status"), "defined")
            self.assertLessEqual(entry["lower_exposure"], entry["upper_exposure"] + 1e-9)
            self.assertGreaterEqual(entry["lift_width"], -1e-9)
            self.assertAlmostEqual(entry["reference_lift"],
                                   reference[entry["rule"]]["log_lift"], places=9)
            self.assertLessEqual(entry["lower_lift"], entry["reference_lift"] + 1e-7)
            self.assertGreaterEqual(entry["upper_lift"], entry["reference_lift"] - 1e-7)

    def test_reference_weights_and_summary_are_consistent(self):
        result = feasible_lift_bounds(self.report, self.weights)
        self.assertEqual(set(result["reference_weights"]), set(self.rule_data))
        self.assertAlmostEqual(sum(result["reference_weights"].values()), 1.0)
        summary = result["summary"]
        if summary["defined_rule_count"]:
            self.assertIsNotNone(summary["median_lift_width"])
            self.assertGreaterEqual(summary["median_lift_width"], 0.0)
            self.assertLessEqual(summary["min_lower_lift"], summary["max_upper_lift"])
        corner = result["positive_corner"]
        if corner is not None:
            self.assertAlmostEqual(sum(corner["weights"].values()), 1.0)
            self.assertGreaterEqual(corner["max_weight"], 1 / len(self.rule_data))
            self.assertGreater(corner["lift"], 0.0)

    def test_min_weight_floor_must_contain_reference(self):
        with self.assertRaisesRegex(ValueError, "minimum-weight floor"):
            feasible_lift_bounds(self.report, self.weights, min_weight=0.4)
        with self.assertRaises(ValueError):
            feasible_lift_bounds(self.report, self.weights, min_weight=float("nan"))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            weights = Path(folder) / "weights.json"
            output = Path(folder) / "bounds.json"
            report.write_text(json.dumps(self.report))
            weights.write_text(json.dumps(self.weights))
            subprocess.run([sys.executable, "-m", "exposure.design_bounds",
                            "--report", str(report), "--weights", str(weights),
                            "--output", str(output)],
                           check=True, capture_output=True, text=True)
            data = json.loads(output.read_text())
            self.assertEqual(data["summary"]["rule_count"], len(self.rule_data))
            self.assertIn("note", data)


if __name__ == "__main__":
    unittest.main()
