import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exposure.design import solve
from exposure.measure import estimate, load_rules


class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule_data = json.loads(Path("examples/exposure_rules.json").read_text())
        cls.report = estimate(load_rules(cls.rule_data), samples=60, seed=42, include_subsets=True)
        cls.report["rules"] = cls.rule_data

    def test_feasible_simplex_and_bound(self):
        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "predicate",
                                           "predicate_indices": [0], "min": 0.5}]}
        result = solve(self.report, specification)
        self.assertEqual(result["status"], "feasible")
        self.assertAlmostEqual(sum(result["weights"].values()), 1)
        self.assertGreaterEqual(result["constraints"][0]["exposure"], 0.5 - 1e-7)
        self.assertLessEqual(result["max_violation"], 1e-7)

    def test_shuffled_columns_preserve_name_alignment(self):
        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "predicate",
                                           "predicate_indices": [0], "min": 0.5}]}
        shuffled = copy.deepcopy(self.report)
        shuffled["columns"].reverse()
        self.assertEqual(solve(self.report, specification)["weights"], solve(shuffled, specification)["weights"])

    def test_infeasible_constraints_have_no_weights(self):
        base = {"rule": "red", "kind": "rule", "predicate_indices": []}
        specification = {"constraints": [{**base, "name": "low", "max": 0.1},
                                          {**base, "name": "high", "min": 0.9}]}
        result = solve(self.report, specification)
        self.assertEqual(result["status"], "infeasible")
        self.assertNotIn("weights", result)

    def test_bad_indices_bounds_and_report(self):
        entry = {"name": "x", "rule": "red", "kind": "pair", "predicate_indices": [0, 0], "min": 0.1}
        with self.assertRaises(ValueError):
            solve(self.report, {"constraints": [entry]})
        entry.update(kind="predicate", predicate_indices=[0], min=float("nan"))
        with self.assertRaises(ValueError):
            solve(self.report, {"constraints": [entry]})
        report = copy.deepcopy(self.report)
        report["include_subsets"] = False
        with self.assertRaises(ValueError):
            solve(report, {"constraints": [entry]})

    def test_solver_failure_is_not_infeasibility(self):
        class Failed:
            status = 4
            success = False
            message = "numerical failure"
        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "rule",
                                           "predicate_indices": [], "min": 0}]}
        with patch("exposure.design.linprog", return_value=Failed()), self.assertRaises(RuntimeError):
            solve(self.report, specification)

    def test_cli_with_pair_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report, constraints, output = folder / "report.json", folder / "constraints.json", folder / "out.json"
            report.write_text(json.dumps(self.report))
            constraints.write_text(json.dumps({"constraints": [{"name": "pair", "rule": "red_left_of_square",
                "kind": "pair", "predicate_indices": [0, 1], "min": 0.2}]}))
            subprocess.run([sys.executable, "-m", "exposure.design", "--report", str(report),
                            "--constraints", str(constraints), "--output", str(output)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text())["status"], "feasible")


if __name__ == "__main__":
    unittest.main()
