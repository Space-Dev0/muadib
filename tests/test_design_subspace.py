import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exposure.design_subspace import diagnose
from exposure.measure import estimate, load_rules


class DesignSubspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def red(variable):
            return {"type": "attribute", "variable": variable, "field": "color", "value": "red"}
        cls.rule_data = {
            "red_a": [red("a")],
            "red_b": [red("b")],
            "blue_a": [{"type": "attribute", "variable": "a", "field": "color", "value": "blue"}],
        }
        cls.report = estimate(load_rules(cls.rule_data), samples=12, seed=42, include_subsets=True)
        cls.report["rules"] = cls.rule_data
        # A known rank-two toy matrix; each repeated red event has the same row.
        for column in cls.report["columns"]:
            red_present = column["condition"] != "blue_a"
            for name in cls.rule_data:
                present = red_present if name.startswith("red") else not red_present
                count = column["accepted_scenes"] if present else 0
                for cell in (column["measurements"][name], column["predicate_subsets"][name][0]):
                    cell["successes"] = count
                    cell["probability"] = count / column["accepted_scenes"]
        cls.weights = {"red_a": 0.5, "red_b": 0.5, "blue_a": 0.0}

    def test_rank_nullspace_and_boundary(self):
        result = diagnose(self.report, self.weights)
        self.assertEqual(result["column_names"], list(self.rule_data))
        self.assertEqual((result["rank"], result["nullity"]), (2, 1))
        self.assertEqual(len(result["singleton_events"]), 2)
        self.assertEqual(result["minimum_reference_weight"], 0.0)
        basis = np.asarray(result["nullspace_basis"])
        matrix = np.asarray([result["simplex_row"]] +
                            [event["column_probabilities"] for event in result["singleton_events"]])
        np.testing.assert_allclose(matrix @ basis.T, 0, atol=1e-12)
        np.testing.assert_allclose(basis @ basis.T, np.eye(1), atol=1e-12)
        self.assertTrue(any("zero-weight boundaries" in text for text in result["caveats"]))

    def test_repeated_singleton_mismatch_rejected(self):
        report = copy.deepcopy(self.report)
        cell = report["columns"][2]["predicate_subsets"]["red_b"][0]
        cell["successes"], cell["probability"] = 12, 1.0
        with self.assertRaisesRegex(ValueError, "Inconsistent singleton"):
            diagnose(report, self.weights)

    def test_invalid_report_and_weights(self):
        report = copy.deepcopy(self.report)
        report["include_subsets"] = False
        with self.assertRaises(ValueError):
            diagnose(report, self.weights)
        with self.assertRaises(ValueError):
            diagnose(self.report, {"red_a": 1.0})

    def test_determinism_and_tolerance(self):
        self.assertEqual(diagnose(self.report, self.weights), diagnose(self.report, self.weights))
        result = diagnose(self.report, self.weights)
        self.assertGreater(result["rank_tolerance"], 0)
        self.assertEqual(result["rank_tolerance_method"], "max(shape)*eps*largest_singular_value")
        self.assertEqual(result["rank"], 2)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            report, weights, out = (directory / x for x in ("report.json", "weights.json", "result.json"))
            report.write_text(json.dumps(self.report), encoding="utf-8")
            weights.write_text(json.dumps(self.weights), encoding="utf-8")
            subprocess.run([sys.executable, "-m", "exposure.design_subspace", "--report", str(report),
                            "--weights", str(weights), "--output", str(out)],
                           check=True, capture_output=True, text=True)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["nullity"], 1)
            self.assertTrue(math.isfinite(result["rank_tolerance"]))


if __name__ == "__main__":
    unittest.main()
