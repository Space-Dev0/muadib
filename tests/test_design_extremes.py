import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.design_bounds import feasible_lift_bounds
from exposure.design_extremes import generate
from exposure.design_subspace import diagnose
from exposure.measure import estimate, load_rules
from exposure.mixture import summarize


class ExtremeDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rules = json.loads(Path("examples/exposure_rules.json").read_text())
        cls.report = estimate(load_rules(rules), samples=60, seed=423, include_subsets=True)
        cls.report["rules"] = rules
        cls.reference = {name: 1 / len(rules) for name in rules}

    def test_endpoints_satisfy_equalities_and_reach_rule_bounds(self):
        result = generate(self.report, self.reference, min_weight=0.01)
        names = list(self.reference)
        self.assertEqual(len(result["designs"]), 1 + 2 * len(names))
        constraints = diagnose(self.report, self.reference)["singleton_events"]
        bounds = {r["rule"]: r for r in feasible_lift_bounds(self.report, self.reference, min_weight=0.01)["rules"]}
        for i, rule in enumerate(names):
            for offset, key in ((1, "lower_exposure"), (2, "upper_exposure")):
                weights = result["designs"][2 * i + offset]["weights"]
                self.assertAlmostEqual(sum(weights.values()), 1.0)
                self.assertGreaterEqual(min(weights.values()), 0.01 - 1e-8)
                for event in constraints:
                    actual = sum(weights[n] * p for n, p in zip(names, event["column_probabilities"], strict=True))
                    self.assertAlmostEqual(actual, event["reference_exposure"], places=7)
                self.assertAlmostEqual(summarize(self.report, weights)["results"][rule]["rule_exposure"],
                                       bounds[rule][key], places=7)

    def test_invalid_floor_and_missing_subsets(self):
        for floor in (float("nan"), -0.1, 0.4, True):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                generate(self.report, self.reference, min_weight=floor)
        malformed = dict(self.report, include_subsets=False)
        with self.assertRaises(ValueError):
            generate(malformed, self.reference)

    def test_cli_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            reference = Path(folder) / "reference.json"
            target = Path(folder) / "designs.json"
            report.write_text(json.dumps(self.report))
            reference.write_text(json.dumps(self.reference))
            command = [sys.executable, "-m", "exposure.design_extremes", "--report", str(report),
                       "--reference", str(reference), "--output", str(target)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            before = target.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
