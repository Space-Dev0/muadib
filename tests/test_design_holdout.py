import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.design_holdout import compare
from exposure.measure import estimate, load_rules


class DesignHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = {
            color: [{"type": "attribute", "variable": "a", "field": "color", "value": color}]
            for color in ("red", "blue", "green")
        }
        cls.training = estimate(load_rules(cls.rules), samples=12, seed=7, include_subsets=True)
        cls.holdout = estimate(load_rules(cls.rules), samples=12, seed=8, include_subsets=True)
        cls.training["rules"] = cls.holdout["rules"] = cls.rules
        cls.reference = dict.fromkeys(cls.rules, 1 / 3)
        cls.designs = {"designs": [
            {"name": color, "weights": {name: float(name == color) for name in cls.rules}}
            for color in cls.rules
        ]}

    def test_fixed_design_deviations_and_diagnostics(self):
        result = compare(self.training, self.holdout, self.reference, self.designs)
        self.assertEqual(result["gate_status"], "not_assessed")
        self.assertEqual(len(result["singleton_deviations"]), 3)
        row = result["singleton_deviations"][0]
        self.assertAlmostEqual(row["max_abs_holdout_deviation"],
                               max(abs(event["holdout_deviation"]) for event in row["events"]))
        self.assertEqual(result["training"]["row_count"], 9)
        self.assertEqual(result["holdout"]["row_count"], 9)

    def test_rejects_same_seed_and_mismatched_rules(self):
        with self.assertRaisesRegex(ValueError, "seeds must differ"):
            compare(self.training, self.training, self.reference, self.designs)
        mismatch = copy.deepcopy(self.holdout)
        mismatch["rules"]["red"][0]["value"] = "yellow"
        with self.assertRaisesRegex(ValueError, "identical rules"):
            compare(self.training, mismatch, self.reference, self.designs)

    def test_cli_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"input-{i}.json" for i in range(4)]
            for path, value in zip(paths, (self.training, self.holdout, self.reference, self.designs)):
                path.write_text(json.dumps(value), encoding="utf-8")
            output = root / "result.json"
            output.write_text("preserve", encoding="utf-8")
            run = subprocess.run([sys.executable, "-m", "exposure.design_holdout",
                                  "--training", str(paths[0]), "--holdout", str(paths[1]),
                                  "--reference", str(paths[2]), "--designs", str(paths[3]),
                                  "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
