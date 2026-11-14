import unittest

from exposure.rules import satisfies
from exposure.spatial_counterfactual_audit import cases


class SpatialCounterfactualTests(unittest.TestCase):
    def test_annotations_make_ties_negative_and_bind_order(self):
        seen = set()
        for kind, delta, shapes, objects, rules in cases():
            if shapes != ('square', 'square', 'square'):
                continue
            self.assertEqual(satisfies(objects, rules['singleton']), delta != 0)
            self.assertEqual(satisfies(objects, rules['bound']), delta > 0)
            seen.add((kind, delta))
        self.assertEqual(len(seen), 14)


if __name__ == '__main__':
    unittest.main()
