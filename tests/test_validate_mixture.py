import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exposure.measure import estimate, load_rules
from exposure.mixture import summarize
from exposure.validate_mixture import validate


class MixtureValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule_data = json.loads(Path("examples/exposure_rules.json").read_text())
        cls.report = estimate(load_rules(cls.rule_data), samples=30, seed=42, include_subsets=True)
        cls.report["rules"] = cls.rule_data
        cls.weights = {name: 0.25 for name in cls.rule_data}

    def test_known_condition_and_independent_sampling(self):
        weights = {name: float(name == "red") for name in self.rule_data}
        result = validate(self.report, weights, samples=40, seed=43)
        self.assertEqual(result["condition_counts"]["red"], 40)
        self.assertEqual(sum(result["condition_counts"].values()), 40)
        self.assertEqual(result["results"]["red"]["successes"], 40)
        self.assertEqual(result["results"]["red"]["predicted_probability"], 1.0)
        self.assertEqual(result["results"]["red"]["observed_minus_predicted"], 0.0)
        self.assertEqual(result, validate(self.report, weights, samples=40, seed=43))
        self.assertEqual(result["rules"], self.rule_data)

    def test_full_mixture_matches_weighted_predictions_and_reports_observation_interval(self):
        result = validate(self.report, self.weights, samples=80, seed=101)
        predicted = summarize(self.report, self.weights)["results"]
        for name, entry in result["results"].items():
            self.assertAlmostEqual(entry["predicted_probability"], predicted[name]["rule_exposure"])
            self.assertAlmostEqual(entry["observed_probability"], entry["successes"] / 80)
            self.assertAlmostEqual(entry["observed_minus_predicted"],
                                   entry["observed_probability"] - entry["predicted_probability"])
            self.assertLessEqual(entry["observed_wilson_95"][0], entry["observed_probability"])
            self.assertGreaterEqual(entry["observed_wilson_95"][1], entry["observed_probability"])
            self.assertNotIn("prediction_weighted_pointwise_wilson_envelope", entry)
        self.assertGreaterEqual(result["total_attempts"], 80)

    def test_rejects_invalid_or_nonindependent_configuration(self):
        for kwargs in ({"samples": 0, "seed": 43}, {"samples": 10, "seed": 42},
                       {"samples": 10, "seed": 43, "max_attempts": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate(self.report, self.weights, **kwargs)
        with self.assertRaises(ValueError):
            validate(self.report, {}, samples=10, seed=43)
        with patch("exposure.validate_mixture.sample_conditioned", side_effect=ValueError("not found")):
            with self.assertRaisesRegex(ValueError, "sample 1"):
                validate(self.report, self.weights, samples=10, seed=43)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report_path, output = folder / "report.json", folder / "check.json"
            report_path.write_text(json.dumps(self.report))
            subprocess.run([sys.executable, "-m", "exposure.validate_mixture", "--report", str(report_path),
                            "--weights", "examples/uniform_weights.json", "--samples", "15", "--seed", "43",
                            "--output", str(output)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text())["validation_samples"], 15)


if __name__ == "__main__":
    unittest.main()
