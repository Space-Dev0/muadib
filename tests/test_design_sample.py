import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.design_sample import sample_designs
from exposure.measure import estimate, load_rules
from exposure.mixture import summarize


class DesignSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def attr(name, color):
            return {"type": "attribute", "variable": name, "field": "color", "value": color}
        cls.rules = {"red_a": [attr("a", "red")], "red_b": [attr("b", "red")],
                     "blue_a": [attr("a", "blue")]}
        cls.report = estimate(load_rules(cls.rules), samples=12, seed=42, include_subsets=True)
        cls.report["rules"] = cls.rules
        for column in cls.report["columns"]:
            red_present = column["condition"] != "blue_a"
            for name in cls.rules:
                present = red_present if name.startswith("red") else not red_present
                count = column["accepted_scenes"] if present else 0
                for cell in (column["measurements"][name], column["predicate_subsets"][name][0]):
                    cell["successes"], cell["probability"] = count, count / column["accepted_scenes"]
        cls.weights = {"red_a": 0.4, "red_b": 0.4, "blue_a": 0.2}

    def test_reproducible_weights_and_marginal_invariance(self):
        options = {"samples": 20, "seed": 7, "min_weight": 0.05, "burn_in": 4, "thin": 3}
        result = sample_designs(self.report, self.weights, **options)
        self.assertEqual(result, sample_designs(self.report, self.weights, **options))
        self.assertEqual(set(result), {"designs"})
        self.assertEqual(len(result["designs"]), 20)
        reference = summarize(self.report, self.weights)["results"]
        for i, design in enumerate(result["designs"], 1):
            self.assertEqual(design["name"], f"sample_{i:04d}")
            weights = design["weights"]
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=7)
            self.assertTrue(all(w >= 0.05 - 1e-7 for w in weights.values()))
            candidate = summarize(self.report, weights)["results"]
            for name in self.rules:
                self.assertAlmostEqual(candidate[name]["predicate_marginals"][0],
                                       reference[name]["predicate_marginals"][0], places=7)

    def test_nullity_zero_and_boundary_rejected(self):
        one_rule = {"red": self.rules["red_a"]}
        report = estimate(load_rules(one_rule), samples=4, seed=1, include_subsets=True)
        report["rules"] = one_rule
        with self.assertRaisesRegex(ValueError, "nullity is zero"):
            sample_designs(report, {"red": 1.0}, samples=1, seed=0)
        boundary = {"red_a": 0.5, "red_b": 0.5, "blue_a": 0.0}
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            sample_designs(self.report, boundary, samples=1, seed=0)
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            sample_designs(self.report, self.weights, samples=1, seed=0, min_weight=0.2)

    def test_invalid_arguments_and_report(self):
        for kwargs in ({"samples": 0}, {"seed": -1}, {"burn_in": -1}, {"thin": 0},
                       {"min_weight": float("nan")}, {"min_weight": True}):
            options = {"samples": 1, "seed": 0, **kwargs}
            with self.subTest(options=options), self.assertRaises(ValueError):
                sample_designs(self.report, self.weights, **options)
        report = copy.deepcopy(self.report)
        report["include_subsets"] = False
        with self.assertRaises(ValueError):
            sample_designs(report, self.weights, samples=1, seed=0)

    def test_cli_and_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            report, weights, out = (directory / x for x in ("report.json", "weights.json", "designs.json"))
            report.write_text(json.dumps(self.report), encoding="utf-8")
            weights.write_text(json.dumps(self.weights), encoding="utf-8")
            command = [sys.executable, "-m", "exposure.design_sample", "--report", str(report),
                       "--weights", str(weights), "--samples", "2", "--seed", "2",
                       "--burn-in", "1", "--thin", "1", "--output", str(out)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            saved = out.read_text(encoding="utf-8")
            self.assertEqual(len(json.loads(saved)["designs"]), 2)
            run = subprocess.run(command + ["--min-weight", "0.2"], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(out.read_text(encoding="utf-8"), saved)


if __name__ == "__main__":
    unittest.main()
