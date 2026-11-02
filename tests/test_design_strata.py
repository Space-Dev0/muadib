import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.design_strata import summarize


class DesignStrataTests(unittest.TestCase):
    def test_rule_centering_and_constant_stratum(self):
        rows = []
        for rule, offset in (("a", 0), ("b", 10)):
            for design, value in (("one", -1), ("two", 1)):
                rows.append({"design": design, "rule": rule, "k_attr": 1, "k_rel": 0,
                             "log_mean_marginal": offset + value, "log_lift": offset - value})
        result = summarize({"rows": rows, "excluded_count": 0})
        self.assertAlmostEqual(result["within_rule_correlation"], -1.0)
        self.assertAlmostEqual(result["row_weighted_rule_mean_correlation"], 1.0)
        self.assertGreater(result["pooled_correlation"], 0)
        constant = [{**row, "log_mean_marginal": 4.0} for row in rows]
        result = summarize({"rows": constant})
        self.assertIsNone(result["within_rule_correlation"])
        self.assertIsNone(result["strata"][0]["correlation"])

    def test_invalid_duplicate_and_nonfinite_rows(self):
        row = {"design": "a", "rule": "r", "k_attr": 1, "k_rel": 0,
               "log_mean_marginal": 1.0, "log_lift": 0.0}
        for rows in ([row, row], [{**row, "log_lift": float("nan")}],
                     [row, {**row, "design": "b", "k_rel": 2}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize({"rows": rows})

    def test_cli_does_not_overwrite_input(self):
        row = {"design": "a", "rule": "r", "k_attr": 1, "k_rel": 0,
               "log_mean_marginal": 1.0, "log_lift": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            content = json.dumps({"rows": [row]})
            path.write_text(content)
            result = subprocess.run([sys.executable, "-m", "exposure.design_strata",
                                     "--report", str(path), "--output", str(path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_text(), content)


if __name__ == "__main__":
    unittest.main()
