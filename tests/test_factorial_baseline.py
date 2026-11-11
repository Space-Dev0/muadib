import unittest

from exposure.factorial_baseline import estimate


class FactorialBaselineTests(unittest.TestCase):
    def test_seeded_counts_and_shared_relation_event(self):
        result = estimate(100, 123)
        self.assertEqual(result, estimate(100, 123))
        self.assertEqual(len(result["rules"]), 24)
        for name, row in result["rules"].items():
            relation = "left_of" if "left_of" in name else "above"
            self.assertLessEqual(row["successes"], result["relation_singletons"][relation]["successes"])
            self.assertLessEqual(row["wilson_95"][0], row["probability"])
            self.assertGreaterEqual(row["wilson_95"][1], row["probability"])

    def test_invalid_counts(self):
        for samples, seed in [(0, 1), (True, 1), (1, False)]:
            with self.assertRaises(ValueError):
                estimate(samples, seed)


if __name__ == "__main__":
    unittest.main()
