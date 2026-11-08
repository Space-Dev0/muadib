import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exposure.design_chains import effective_sample_size, sample_chains, split_rhat
from exposure.measure import estimate, load_rules


def _report():
    def attr(name, color):
        return {"type": "attribute", "variable": name, "field": "color", "value": color}

    rules = {"red_a": [attr("a", "red")], "red_b": [attr("b", "red")],
             "blue_a": [attr("a", "blue")]}
    report = estimate(load_rules(rules), samples=12, seed=42, include_subsets=True)
    report["rules"] = rules
    for column in report["columns"]:
        red_present = column["condition"] != "blue_a"
        for name in rules:
            present = red_present if name.startswith("red") else not red_present
            count = column["accepted_scenes"] if present else 0
            for cell in (column["measurements"][name],
                         column["predicate_subsets"][name][0]):
                cell["successes"], cell["probability"] = count, count / column["accepted_scenes"]
    return report


class DesignChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _report()
        cls.weights = {"red_a": 0.4, "red_b": 0.4, "blue_a": 0.2}

    def test_diagnostics_are_reproducible_and_complete(self):
        options = {"chains": 2, "samples": 6, "seed": 3, "min_weight": 0.05,
                   "burn_in": 2, "thin": 1}
        result = sample_chains(self.report, self.weights, **options)
        self.assertEqual(result, sample_chains(self.report, self.weights, **options))
        self.assertEqual(result["chains"], 2)
        self.assertEqual(len(result["chain_results"]), 2)
        self.assertEqual([chain["seed"] for chain in result["chain_results"]], [3, 4])
        summary = result["diagnostics"]["summary"]
        self.assertEqual(summary["rule_count"], 3)
        self.assertIn(summary["status"], {"pass", "fail", "insufficient"})
        for name in self.weights:
            self.assertIn(name, result["diagnostics"]["rules"])
            entry = result["diagnostics"]["rules"][name]
            self.assertTrue(entry["finite"])
            self.assertIsNotNone(entry["range"])

    def test_split_rhat_and_ess(self):
        first = np.arange(10, dtype=float)
        second = first + 0.5
        self.assertIsNotNone(split_rhat([first, second]))
        self.assertIsNone(split_rhat([first]))
        self.assertIsNone(split_rhat([np.arange(3, dtype=float)]))
        self.assertIsNotNone(effective_sample_size(np.arange(20, dtype=float)))
        self.assertIsNone(effective_sample_size(np.array([1.0, 1.0, 1.0])))
        self.assertIsNone(effective_sample_size(np.array([float('nan'), 1, 2, 3, 4])))
        independent = np.random.default_rng(21).normal(size=5000)
        self.assertGreater(effective_sample_size(independent), 2500)

    def test_invalid_chains(self):
        for chains in (0, -1, True, 1.5):
            with self.subTest(chains=chains), self.assertRaises(ValueError):
                sample_chains(self.report, self.weights, chains=chains, samples=1, seed=0)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            weights = Path(folder) / "weights.json"
            output = Path(folder) / "chains.json"
            report.write_text(json.dumps(self.report))
            weights.write_text(json.dumps(self.weights))
            subprocess.run([sys.executable, "-m", "exposure.design_chains",
                            "--report", str(report), "--weights", str(weights),
                            "--chains", "2", "--samples", "2", "--seed", "5",
                            "--burn-in", "1", "--thin", "1", "--output", str(output)],
                           check=True, capture_output=True, text=True)
            data = json.loads(output.read_text())
            self.assertEqual(data["chains"], 2)
            self.assertEqual(len(data["chain_results"]), 2)


if __name__ == "__main__":
    unittest.main()
