import copy
import unittest

from exposure.sector_benchmark import generate, rule_inventory
from exposure.sector_benchmark_audit import audit


class SectorBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = generate(train_seed=112, heldout_seed=113, scenes_per_rule=1)

    def test_complete_counterbalanced_manifest(self):
        result = audit(self.manifest)
        self.assertEqual(len(rule_inventory()), 288)
        self.assertEqual(result["scenes"], 576)
        self.assertEqual(result["direction_counts"], {"left": 288, "above": 288})
        self.assertEqual(result["exact_pixel_duplicates"], 0)

    def test_reject_tampering_and_missing_data(self):
        for mutate in (lambda m: m["records"].pop(),
                       lambda m: m["records"][0].update(pixel_sha256="0" * 64),
                       lambda m: m["records"].append(m["records"][0]),
                       lambda m: m["parameters"].update(max_distance=100)):
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(self.manifest)
                mutate(changed)
                with self.assertRaises(ValueError):
                    audit(changed)

    def test_reject_invalid_seed_or_attempt_budget(self):
        for kwargs in ({"train_seed": 1, "heldout_seed": 1, "scenes_per_rule": 1},
                       {"train_seed": 1, "heldout_seed": 2, "scenes_per_rule": 0}):
            with self.assertRaises(ValueError):
                generate(**kwargs)


if __name__ == "__main__":
    unittest.main()
