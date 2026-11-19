import unittest

from exposure.sector_benchmark import generate
from exposure.sector_checker_audit import run


class SectorCheckerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = generate(train_seed=121, heldout_seed=122, scenes_per_rule=1)

    def test_balanced_pairs_and_unique_pixels(self):
        report = run(self.manifest, seed=123)
        self.assertEqual(len(report["records"]), 576)
        self.assertEqual(len({row["pixel_sha256"] for row in report["records"]}), 576)
        for row in report["counts"].values():
            self.assertEqual(sum(row.values()), 288)

    def test_reject_reused_seed(self):
        with self.assertRaises(ValueError):
            run(self.manifest, seed=121)


if __name__ == "__main__":
    unittest.main()
