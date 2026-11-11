import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.mosaic_contrast import summarize


class MosaicContrastTests(unittest.TestCase):
    def setUp(self):
        self.data = {"runs": [
            {"seed": seed, "run_id": f"{seed}-{complexity}-{imbalance}",
             "complexity": complexity, "imbalance": imbalance,
             "successes": successes, "total": 10}
            for seed in (1, 2)
            for complexity, imbalance, successes in (
                ("low", "balanced", 9), ("low", "imbalanced", 8),
                ("high", "balanced", 5), ("high", "imbalanced", 4))
        ]}

    def test_matched_contrast(self):
        report = summarize(self.data)
        self.assertEqual(report["seed_count"], 2)
        self.assertAlmostEqual(report["mean"]["complexity_drop"], 0.4)
        self.assertAlmostEqual(report["mean"]["imbalance_drop"], 0.1)
        self.assertAlmostEqual(report["mean"]["difference_in_drops"], 0.3)

    def test_incomplete_duplicate_or_invalid_rows_rejected(self):
        cases = []
        missing = copy.deepcopy(self.data); missing["runs"].pop(); cases.append(missing)
        repeated = copy.deepcopy(self.data); repeated["runs"].append(copy.deepcopy(repeated["runs"][0])); cases.append(repeated)
        bad = copy.deepcopy(self.data); bad["runs"][0]["successes"] = 11; cases.append(bad)
        bad_bool = copy.deepcopy(self.data); bad_bool["runs"][0]["total"] = True; cases.append(bad_bool)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                summarize(case)

    def test_seed_level_variation_and_single_block(self):
        data = copy.deepcopy(self.data)
        data["runs"][4]["successes"] = 7
        report = summarize(data)
        self.assertAlmostEqual(report["between_seed_variation"]["complexity_drop"]["sample_sd"], 0.1 / (2 ** 0.5))
        self.assertAlmostEqual(report["between_seed_variation"]["complexity_drop"]["standard_error"], 0.05)
        one = summarize({"runs": data["runs"][:4]})
        self.assertIsNone(one["between_seed_variation"]["complexity_drop"]["standard_error"])

    def test_cli_output_never_overwrites_input_or_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "runs.json"
            output = Path(directory) / "report.json"
            original = json.dumps(self.data)
            source.write_text(original, encoding="utf-8")
            command = [sys.executable, "-m", "exposure.mosaic_contrast", "--input", str(source), "--output"]
            subprocess.run(command + [str(output)], check=True, capture_output=True, text=True)
            first = output.read_text(encoding="utf-8")
            self.assertNotEqual(subprocess.run(command + [str(output)], capture_output=True).returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), first)
            self.assertNotEqual(subprocess.run(command + [str(source)], capture_output=True).returncode, 0)
            self.assertEqual(source.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
