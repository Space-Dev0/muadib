import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.design_identifiability import inspect


def report(*, constant_relation=False, fixed_marginal=False):
    rows = []
    for rule in range(4):
        for design in range(4):
            rows.append({"rule": f"r{rule}", "design": f"d{design}",
                         "k_attr": 1, "k_rel": 1 if constant_relation else rule % 2,
                         "log_mean_marginal": rule + (0 if fixed_marginal else (design - 1.5) ** 2),
                         "log_lift": (design - 1.5) * (rule + 1)})
    return {"rows": rows, "excluded_count": 0}


class IdentifiabilityTests(unittest.TestCase):
    def test_full_rank_features(self):
        out = inspect(report())
        self.assertEqual(out["rank"], 3)
        self.assertTrue(all(feature["status"] == "identified" for feature in out["features"].values()))

    def test_rule_intercepts_absorb_fixed_marginal(self):
        out = inspect(report(fixed_marginal=True))
        self.assertEqual(out["rank"], 2)
        self.assertEqual(out["features"]["log_mean_marginal"]["status"], "no_within_rule_variation")
        self.assertEqual(out["features"]["log_lift"]["status"], "identified")
        self.assertEqual(out["features"]["log_lift_x_k_rel"]["status"], "identified")
        v = out["lift_variance"]
        self.assertAlmostEqual(v["total"], v["within_rule"] + v["between_rule"])
        self.assertEqual(out["gate_status"], "not_assessed")

    def test_interaction_collinearity_and_identified_marginal(self):
        out = inspect(report(constant_relation=True))
        self.assertEqual(out["rank"], 2)
        self.assertEqual(out["features"]["log_mean_marginal"]["status"], "identified")
        self.assertEqual(out["features"]["log_lift"]["status"], "collinear")
        self.assertEqual(out["features"]["log_lift_x_k_rel"]["status"], "collinear")

    def test_reject_duplicates_and_nonfinite_interaction(self):
        obj = report()
        obj["rows"].append(obj["rows"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inspect(obj)
        obj = report()
        obj["rows"][0]["log_lift"] = 1e308
        for row in obj["rows"][:4]:
            row["k_rel"] = 10
        with self.assertRaisesRegex(ValueError, "interactions must be finite"):
            inspect(obj)

    def test_cli_will_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/"input.json", Path(folder)/"output.json"
            source.write_text(json.dumps(report()))
            target.write_text("keep")
            result = subprocess.run([sys.executable, "-m", "exposure.design_identifiability", "--report", str(source), "--output", str(target)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
