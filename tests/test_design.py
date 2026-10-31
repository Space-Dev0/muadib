import copy
import json
import math
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

    def test_positive_min_weight_floor(self):
        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "rule",
                                           "predicate_indices": [], "min": 0.0}]}
        result = solve(self.report, specification, min_weight=0.05)
        self.assertEqual(result["status"], "feasible")
        self.assertEqual(result["min_weight"], 0.05)
        self.assertTrue(all(weight >= 0.05 - 1e-7 for weight in result["weights"].values()))
        self.assertLessEqual(result["max_violation"], 1e-7)

    def test_floor_violation_is_detected(self):
        class BelowFloor:
            status = 0
            success = True
            x = [0.01, 0.01, 0.01, 0.97]
            message = ""

        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "rule",
                                           "predicate_indices": [], "min": 0.0}]}
        with patch("exposure.design.linprog", return_value=BelowFloor()):
            with self.assertRaisesRegex(RuntimeError, "constraint violation"):
                solve(self.report, specification, min_weight=0.05)

    def test_min_weight_floor_above_simplex_is_infeasible(self):
        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "rule",
                                           "predicate_indices": [], "min": 0.0}]}
        result = solve(self.report, specification, min_weight=0.3)
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["min_weight"], 0.3)
        self.assertNotIn("weights", result)

    def test_invalid_min_weight_values(self):
        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "rule",
                                           "predicate_indices": [], "min": 0.0}]}
        for value in (-0.01, float("nan"), float("inf"), True, "0.1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                solve(self.report, specification, min_weight=value)

    def test_zero_min_weight_matches_default(self):
        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "rule",
                                           "predicate_indices": [], "min": 0.0}]}
        default = solve(self.report, specification)
        explicit = solve(self.report, specification, min_weight=0.0)
        self.assertEqual(default["status"], "feasible")
        self.assertEqual(explicit["status"], "feasible")
        self.assertEqual(default["min_weight"], 0.0)
        self.assertEqual(explicit["min_weight"], 0.0)
        for name, weight in default["weights"].items():
            self.assertAlmostEqual(weight, explicit["weights"][name])

    def test_cli_min_weight_option(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            report_path = directory / "report.json"
            constraints_path = directory / "constraints.json"
            output = directory / "design.json"
            report_path.write_text(json.dumps(self.report))
            constraints_path.write_text(json.dumps({"constraints": [
                {"name": "x", "rule": "red", "kind": "rule",
                 "predicate_indices": [], "min": 0.0}
            ]}))
            subprocess.run([sys.executable, "-m", "exposure.design", "--report", str(report_path),
                            "--constraints", str(constraints_path), "--min-weight", "0.05",
                            "--output", str(output)], check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "feasible")
            self.assertEqual(result["min_weight"], 0.05)
            self.assertTrue(all(weight >= 0.05 - 1e-7 for weight in result["weights"].values()))

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

    def test_min_kl_uniform_and_analytical_optimum(self):
        report = copy.deepcopy(self.report)
        for column in report["columns"]:
            count = column["accepted_scenes"] if column["condition"] == "red" else 0
            for cell in (column["measurements"]["red"], column["predicate_subsets"]["red"][0]):
                cell["successes"], cell["probability"] = count, count / column["accepted_scenes"]
        entry = {"name": "red", "rule": "red", "kind": "rule", "predicate_indices": [], "min": 0.0}
        result = solve(report, {"constraints": [entry]}, objective="min-kl")
        self.assertTrue(all(abs(weight - 0.25) < 1e-5 for weight in result["weights"].values()))
        self.assertAlmostEqual(result["kl_to_uniform"], 0.0, places=8)
        entry["min"] = 0.6
        result = solve(report, {"constraints": [entry]}, objective="min-kl")
        self.assertAlmostEqual(result["weights"]["red"], 0.6, places=5)
        for name in ("square", "red_left_of_square", "blue_above_circle"):
            self.assertAlmostEqual(result["weights"][name], 0.4 / 3, places=5)
        self.assertLessEqual(result["max_violation"], 1e-7)
        entry["max"] = 0.6
        result = solve(report, {"constraints": [entry]}, objective="min-kl", min_weight=0.1)
        self.assertAlmostEqual(result["weights"]["red"], 0.6, places=5)
        self.assertTrue(all(weight >= 0.1 - 1e-7 for weight in result["weights"].values()))
        self.assertEqual(solve(report, {"constraints": [entry]}, objective="min-kl", min_weight=0.2)["status"], "infeasible")

    def test_min_kl_zero_weight_and_optimizer_failure(self):
        report = copy.deepcopy(self.report)
        for column in report["columns"]:
            count = column["accepted_scenes"] if column["condition"] == "red" else 0
            for cell in (column["measurements"]["red"], column["predicate_subsets"]["red"][0]):
                cell["successes"], cell["probability"] = count, count / column["accepted_scenes"]
        spec = {"constraints": [{"name": "zero", "rule": "red", "kind": "rule",
                                  "predicate_indices": [], "max": 0.0}]}
        result = solve(report, spec, objective="min-kl")
        self.assertAlmostEqual(result["weights"]["red"], 0.0, places=7)
        self.assertAlmostEqual(result["kl_to_uniform"], math.log(4 / 3), places=6)
        class Failed:
            success = False
            status = 9
            message = "iteration limit"
        with patch("exposure.design.minimize", return_value=Failed()), self.assertRaisesRegex(RuntimeError, "optimizer failed"):
            solve(report, spec, objective="min-kl")

    def test_min_kl_redundant_constant_equality(self):
        report = copy.deepcopy(self.report)
        for column in report["columns"]:
            for cell in (column["measurements"]["red"], column["predicate_subsets"]["red"][0]):
                cell["successes"], cell["probability"] = column["accepted_scenes"], 1.0
        spec = {"constraints": [{"name": "always", "rule": "red", "kind": "rule",
                                 "predicate_indices": [], "min": 1.0, "max": 1.0}]}
        result = solve(report, spec, objective="min-kl")
        self.assertTrue(all(abs(weight - 0.25) < 1e-5 for weight in result["weights"].values()))

    def test_cli_with_pair_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report, constraints, output = folder / "report.json", folder / "constraints.json", folder / "out.json"
            report.write_text(json.dumps(self.report))
            constraints.write_text(json.dumps({"constraints": [{"name": "pair", "rule": "red_left_of_square",
                "kind": "pair", "predicate_indices": [0, 1], "min": 0.2}]}))
            subprocess.run([sys.executable, "-m", "exposure.design", "--report", str(report),
                            "--constraints", str(constraints), "--output", str(output), "--objective", "min-kl"],
                           check=True, capture_output=True, text=True)
            data = json.loads(output.read_text())
            self.assertEqual(data["status"], "feasible")
            self.assertEqual(data["objective"], "min-kl")


if __name__ == "__main__":
    unittest.main()
