import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.benchmark_audit import audit
from exposure.templates import build_templates


class BenchmarkAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules, _ = build_templates()
        cls.result = audit(cls.rules)

    def test_analytic_relation_saturation(self):
        relation = self.result["relation_singleton"]
        self.assertAlmostEqual(relation["analytic_exposure"], 1 - (3 / math.comb(9, 3)) / 49)
        self.assertGreater(relation["analytic_exposure"], 0.999)
        self.assertTrue(relation["saturated"])
        self.assertIsNone(relation["simulated_exposure"])

    def test_template_set_is_not_confirmatory_ready(self):
        self.assertFalse(self.result["confirmatory_ready"])
        findings = {item["id"] for item in self.result["blocking_findings"]}
        self.assertIn("saturated_relation_singletons", findings)
        self.assertIn("relation_kind_binding_confound", findings)
        self.assertIn("incomplete_same_object_pair_coverage", findings)
        self.assertIn("no_held_out_template_split", findings)

    def test_endpoint_binding_confound_is_cell_specific(self):
        confounded = self.result["endpoint_binding"]["confounded_cells"]
        cells = {(item["k_attr"], item["k_rel"]) for item in confounded}
        self.assertIn((2, 1), cells)
        by_cell = self.result["endpoint_binding"]["by_cell"]
        left = by_cell["attr2_rel1"]["by_kind"]["left_of"]
        above = by_cell["attr2_rel1"]["by_kind"]["above"]
        self.assertEqual(left, {"subject_only": 3})
        self.assertEqual(above, {"both": 3})

    def test_pair_coverage_reports_same_object_gap(self):
        coverage = self.result["pair_coverage"]
        self.assertEqual(coverage["same_object_pair_count"], 3)
        self.assertEqual(coverage["all_pair_count"], 12)
        self.assertIn(["red", "square"], coverage["missing_same_object"])

    def test_simulation_is_seeded_and_reproducible(self):
        first = audit(self.rules, samples=200, seed=7)
        second = audit(self.rules, samples=200, seed=7)
        self.assertEqual(first["relation_singleton"]["simulated_exposure"],
                         second["relation_singleton"]["simulated_exposure"])
        for value in first["relation_singleton"]["simulated_exposure"].values():
            self.assertGreater(value, 0.99)

    def test_cli_require_confirmatory_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            rules = Path(folder) / "rules.json"
            output = Path(folder) / "audit.json"
            rules.write_text(json.dumps(self.rules))
            command = [sys.executable, "-m", "exposure.benchmark_audit",
                       "--rules", str(rules), "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertFalse(json.loads(output.read_text())["confirmatory_ready"])
            failed = subprocess.run(command + ["--require-confirmatory"],
                                    capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("not confirmatory-ready", failed.stderr)


if __name__ == "__main__":
    unittest.main()
