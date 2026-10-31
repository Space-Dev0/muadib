import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.design_diagnostics import FEATURES, diagnose, statistics
from exposure.measure import estimate, load_rules


class DesignDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = json.loads(Path("examples/exposure_rules.json").read_text())
        cls.report = estimate(load_rules(cls.rules), samples=30, seed=42, include_subsets=True)
        cls.report["rules"] = cls.rules
        cls.designs = {"designs": [{"name": f"d{i}", "weights": {key: (1.0 if j == i else 0.0)
                        for j, key in enumerate(cls.rules)}} for i in range(3)]}

    def test_independent_orthogonal_features_have_unit_vif(self):
        rows = [dict(zip(FEATURES, values, strict=True)) for values in itertools.product((-1, 1), repeat=4)]
        stats = statistics(rows)
        self.assertAlmostEqual(stats["correlations"][FEATURES[0]][FEATURES[1]], 0)
        for feature in FEATURES:
            self.assertEqual(stats["vif"][feature]["status"], "ok")
            self.assertAlmostEqual(stats["vif"][feature]["value"], 1)

    def test_collinear_constant_and_too_few_rows(self):
        rows = [{FEATURES[0]: x, FEATURES[1]: 2 * x, FEATURES[2]: x % 2,
                 FEATURES[3]: 1} for x in range(8)]
        stats = statistics(rows)
        self.assertEqual(stats["vif"][FEATURES[0]]["status"], "collinear_or_near_collinear")
        self.assertEqual(stats["vif"][FEATURES[3]]["status"], "constant")
        self.assertIsNone(stats["correlations"][FEATURES[3]][FEATURES[0]])
        self.assertEqual(statistics(rows[:3])["vif"][FEATURES[0]]["status"], "insufficient_rows")

    def test_undefined_lifts_are_excluded_and_reported(self):
        report = json.loads(json.dumps(self.report))
        report["columns"][0]["measurements"]["red_left_of_square"].update(successes=0, probability=0)
        weights = {key: float(i == 0) for i, key in enumerate(self.rules)}
        config = {"designs": [{"name": str(i), "weights": weights} for i in range(3)]}
        result = diagnose(report, config)
        self.assertTrue(any(item["reason"] == "zero_joint" for item in result["excluded"]))
        self.assertEqual(result["row_count"] + result["excluded_count"], 12)
        self.assertEqual(result["gate_status"], "not_assessed")
        json.dumps(result, allow_nan=False)

    def test_validation_and_cli(self):
        self.assertEqual(diagnose(self.report, self.designs), diagnose(self.report, self.designs))
        for bad in ({"designs": []}, {"designs": self.designs["designs"][:2]},
                    {"designs": [self.designs["designs"][0]] * 3}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                diagnose(self.report, bad)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path, designs_path, output = root / "report.json", root / "designs.json", root / "out.json"
            report_path.write_text(json.dumps(self.report))
            designs_path.write_text(json.dumps(self.designs))
            subprocess.run([sys.executable, "-m", "exposure.design_diagnostics", "--report", str(report_path),
                            "--designs", str(designs_path), "--output", str(output)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text())["gate_status"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
