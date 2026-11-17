import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.design import solve
from exposure.measure import estimate, load_rules
from exposure.mixture import _lift_status, _log_lift, summarize


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.rule_data = json.loads(Path("examples/exposure_rules.json").read_text())
        self.report = estimate(load_rules(self.rule_data), samples=40, seed=42, include_subsets=True)
        self.report["rules"] = self.rule_data
        self.weights = {name: 0.25 for name in self.rule_data}

    def test_weighted_rule_marginals_and_pair_lift(self):
        result = summarize(self.report, self.weights)["results"]["red_left_of_square"]
        columns = self.report["columns"]
        rule = "red_left_of_square"
        expected_rule = sum(c["measurements"][rule]["probability"] for c in columns) / 4
        marginal = sum(c["predicate_subsets"][rule][0]["probability"] for c in columns) / 4
        pair = sum(c["predicate_subsets"][rule][3]["probability"] for c in columns) / 4
        second = sum(c["predicate_subsets"][rule][1]["probability"] for c in columns) / 4
        self.assertAlmostEqual(result["rule_exposure"], expected_rule)
        self.assertAlmostEqual(result["predicate_marginals"][0], marginal)
        self.assertAlmostEqual(result["pairs"][0]["exposure"], pair)
        self.assertAlmostEqual(result["pairs"][0]["log_lift"], math.log(pair / (marginal * second)))

    def test_zero_log_lift_and_validation(self):
        self.assertIsNone(_log_lift(0, [0.5, 0.5]))
        self.assertIsNone(_log_lift(0.5, [0, 0.5]))
        self.assertEqual(_lift_status(0, [0.5, 0.5]), "zero_joint")
        self.assertEqual(_lift_status(0, [0, 0.5]), "zero_marginal")
        self.assertEqual(_lift_status(0.25, [0.5, 0.5]), "defined")
        for weights in ({}, {**self.weights, "red": -1}, {**self.weights, "red": float("nan")},
                        {**self.weights, "red": 0.5}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                summarize(self.report, weights)
        self.report["include_subsets"] = False
        with self.assertRaisesRegex(ValueError, "--include-subsets"):
            summarize(self.report, self.weights)

    def test_roundoff_at_certain_exposure_is_clamped(self):
        atoms = [{'type': 'attribute', 'variable': 'a', 'field': 'color', 'value': 'red'},
                 {'type': 'attribute', 'variable': 'a', 'field': 'shape', 'value': 'circle'}]
        data = {f'r{i}': atoms for i in range(10)}
        report = estimate(load_rules(data), samples=2, seed=7, include_subsets=True)
        report['rules'] = data
        values = [0.15965424664824548, 0.07745534001855994, 0.1367671936038439,
                  0.07854315964607994, 0.048559428329787574, 0.12516888466849205,
                  0.011185625999167349, 0.14057959782930213, 0.17182938488281488,
                  0.0502571383737067]
        self.assertGreater(math.fsum(x / math.fsum(values) for x in values), 1.0)
        result = summarize(report, dict(zip(data, values, strict=True)))['results']['r0']
        self.assertEqual(result['rule_exposure'], 1.0)
        self.assertEqual(result['predicate_marginals'], [1.0, 1.0])
        self.assertEqual(result['pairs'][0]['exposure'], 1.0)

    def test_absent_include_subsets_has_actionable_error(self):
        report = dict(self.report)
        report.pop("include_subsets", None)
        with self.assertRaisesRegex(ValueError, "old or incomplete") as summarize_error:
            summarize(report, self.weights)
        message = str(summarize_error.exception)
        for expected in ("python -m exposure.measure", "--include-subsets", "new --output", "--report"):
            self.assertIn(expected, message)
        specification = {"constraints": [{"name": "x", "rule": "red", "kind": "rule",
                                           "predicate_indices": [], "min": 0.0}]}
        with self.assertRaisesRegex(ValueError, "old or incomplete") as design_error:
            solve(report, specification)
        self.assertIn("--include-subsets", str(design_error.exception))

    def test_incomplete_or_inconsistent_measurements_fail(self):
        entry = self.report["columns"][0]["predicate_subsets"]["red_left_of_square"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize(self.report, self.weights)
        self.report["columns"][0]["predicate_subsets"]["red_left_of_square"].append(entry)
        self.report["columns"][0]["measurements"]["red"]["probability"] = 0.123
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            summarize(self.report, self.weights)

    def test_invalid_conditioned_self_satisfaction_fails(self):
        cell = self.report["columns"][0]["measurements"]["red"]
        cell["successes"], cell["probability"] = 0, 0.0
        with self.assertRaisesRegex(ValueError, "Conditioned scenes"):
            summarize(self.report, self.weights)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            output = Path(directory) / "mixture.json"
            report.write_text(json.dumps(self.report))
            subprocess.run([sys.executable, "-m", "exposure.mixture", "--report", str(report),
                            "--weights", "examples/uniform_weights.json", "--output", str(output)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(set(json.loads(output.read_text())["results"]), set(self.rule_data))


if __name__ == "__main__":
    unittest.main()
