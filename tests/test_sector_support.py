import random
import unittest

from exposure.render import _sample_objects
from exposure.sector_support import measure
from exposure.spatial_sector import exists_sector


class SectorSupportTests(unittest.TestCase):
    def test_matches_independent_existential_count(self):
        samples, seed = 500, 42
        result = measure(samples, seed)
        self.assertEqual(result, measure(samples, seed))
        self.assertEqual(len(result["rows"]), 288)
        row = next(row for row in result["rows"] if row["direction"] == "left"
                   and row["subject"] == ["red", "circle"]
                   and row["target"] == ["blue", "square"])
        rng = random.Random(seed)
        expected = sum(exists_sector(
            _sample_objects(rng), direction="left",
            subject_filter={"color": "red", "shape": "circle"},
            target_filter={"color": "blue", "shape": "square"})
            for _ in range(samples))
        self.assertEqual(row["count"], expected)
        self.assertLessEqual(row["count"], samples)

    def test_invalid_arguments(self):
        for samples, seed in ((0, 1), (1, -1), (True, 1)):
            with self.assertRaises(ValueError):
                measure(samples, seed)


if __name__ == "__main__":
    unittest.main()
