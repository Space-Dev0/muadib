import unittest

from exposure.checker_overlap import paired_scenes, run
from exposure.rules import satisfies


class CheckerOverlapTests(unittest.TestCase):
    def test_paired_geometry_and_balanced_truth(self):
        pairs = list(paired_scenes(12, 20))
        self.assertEqual(sum(satisfies(control, rule) for _, rule, control, _ in pairs), 10)
        for _, rule, control, overlap in pairs:
            self.assertEqual([(o.color, o.shape) for o in control],
                             [(o.color, o.shape) for o in overlap])
            self.assertEqual(satisfies(control, rule), satisfies(overlap, rule))
            self.assertEqual(control[1].x - control[0].x, 20)
            self.assertEqual(overlap[1].x - overlap[0].x, 4)
        self.assertEqual(pairs, list(paired_scenes(12, 20)))

    def test_counts_and_invalid_arguments(self):
        result = run(2, 4)
        for counts in result['counts'].values():
            self.assertEqual(sum(counts.values()), 4)
            self.assertEqual(sum(v for k, v in counts.items() if k.startswith('positive_')), 2)
        for seed, size in ((-1, 2), (1, 3), (True, 2)):
            with self.assertRaises(ValueError):
                list(paired_scenes(seed, size))


if __name__ == '__main__':
    unittest.main()
