import unittest
from collections import Counter

from exposure.factorial_relations import build
from exposure.measure import load_rules
from exposure.render import COLORS, SHAPES


class FactorialRelationsTests(unittest.TestCase):
    def test_all_color_shape_pairs_cross_both_relations_and_bind_endpoints(self):
        definitions = build()
        self.assertEqual(len(definitions), 24)
        self.assertEqual(len(load_rules(definitions)), 24)
        counts = Counter()
        for rule in definitions.values():
            color, shape, relation = rule
            self.assertEqual((color["variable"], shape["variable"]), ("a", "b"))
            self.assertEqual((relation["subject"], relation["object"]), ("a", "b"))
            counts[(color["value"], shape["value"], relation["kind"])] += 1
        self.assertEqual(set(counts), {(c, s, r) for c in COLORS for s in SHAPES
                                       for r in ("left_of", "above")})
        self.assertTrue(all(count == 1 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
