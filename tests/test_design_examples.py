"""Verify that the illustrative matched-marginal designs work on their pilot matrix."""

import json
import unittest
from pathlib import Path

from exposure.design import solve
from exposure.measure import estimate, load_rules


class MatchedMarginalExamplesTests(unittest.TestCase):
    def test_low_and_high_examples_on_documented_pilot(self):
        rules = json.loads(Path("examples/exposure_rules.json").read_text())
        report = estimate(load_rules(rules), samples=200, seed=42, include_subsets=True)
        report["rules"] = rules
        results = []
        for path in ("examples/fixed_marginals_low.json", "examples/fixed_marginals_high.json"):
            result = solve(report, json.loads(Path(path).read_text()), objective="min-kl")
            self.assertEqual(result["status"], "feasible")
            results.append([entry["exposure"] for entry in result["constraints"]])
        self.assertAlmostEqual(results[0][0], results[1][0], places=6)
        self.assertAlmostEqual(results[0][1], results[1][1], places=6)
        self.assertLessEqual(results[0][2], 0.32 + 1e-7)
        self.assertGreaterEqual(results[1][2], 0.55 - 1e-7)


if __name__ == "__main__":
    unittest.main()
