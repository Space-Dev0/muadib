import unittest

from exposure.audit_block_comparison import compare


class SeedBlockTests(unittest.TestCase):
    def test_paired_bootstrap_and_abstentions(self):
        records = []
        for seed in range(4):
            for condition in ("a", "b"):
                records.append({"seed": seed, "condition": condition, "human": True,
                                "standard": None, "candidate": seed % 2 == 0 or None})
        a = compare({"records": records}, bootstrap=200, seed=5)
        self.assertEqual(a, compare({"records": records}, bootstrap=200, seed=5))
        self.assertEqual(a["standard"]["coverage"], 0)
        self.assertEqual(a["candidate"]["coverage"], 0.5)
        self.assertEqual(a["candidate"]["accepted_disagreements"], 0)
        self.assertEqual(a["coverage_difference"], 0.5)
        self.assertLessEqual(a["paired_seed_bootstrap_95"][0], 0.5)
        self.assertGreaterEqual(a["paired_seed_bootstrap_95"][1], 0.5)

    def test_incomplete_and_duplicate_blocks_are_rejected(self):
        row = {"seed": 0, "condition": "a", "human": True, "standard": None, "candidate": False}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare({"records": [row, row]})
        with self.assertRaisesRegex(ValueError, "same conditions"):
            compare({"records": [row, dict(row, seed=0, condition="b"), dict(row, seed=1)]})
        self.assertIsNone(compare({"records": [row]})["paired_seed_bootstrap_95"])


if __name__ == "__main__":
    unittest.main()
