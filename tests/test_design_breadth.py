import random
import unittest
import numpy as np

from exposure.design_breadth import estimate, pair_matrix, rule_log_breadth
from exposure.measure import estimate as measure
from exposure.rules import Attribute, Relation, SceneObject, satisfies


class BreadthTests(unittest.TestCase):
    def test_pair_matrix_matches_existential_semantics(self):
        predicates = (Attribute('a', 'color', 'red'), Attribute('b', 'shape', 'circle'),
                      Relation('a', 'left_of', 'b'), Attribute('a', 'shape', 'square'))
        rng = random.Random(721)
        for _ in range(20):
            scene = tuple(SceneObject(rng.choice(('red', 'blue')), rng.choice(('circle', 'square')),
                                      rng.randrange(3), rng.randrange(3)) for _ in range(3))
            matrix = pair_matrix(scene, predicates)
            for i, p in enumerate(predicates):
                for j, q in enumerate(predicates):
                    self.assertEqual(bool(matrix[i, j]), satisfies(scene, (p, q)))

    def test_reproduces_source_subset_counts(self):
        data = {'red': [{'type':'attribute', 'variable':'a', 'field':'color', 'value':'red'}],
                'blue_circle': [{'type':'attribute', 'variable':'a', 'field':'color', 'value':'blue'},
                                {'type':'attribute', 'variable':'a', 'field':'shape', 'value':'circle'}]}
        from exposure.measure import load_rules
        rules = load_rules(data); source = measure(rules, samples=8, seed=6, include_subsets=True)
        source['rules'] = data
        result = estimate(source)
        self.assertEqual(len(result['counts_by_condition']), 2)
        source['columns'][0]['predicate_subsets']['red'][0]['successes'] += 1
        with self.assertRaisesRegex(ValueError, 'do not match source'):
            estimate(source)

    def test_entropy_and_no_partners(self):
        p = Attribute('a', 'color', 'red'); q = Attribute('a', 'shape', 'circle')
        a = np.array([[[2, 1], [1, 2]]])
        result = rule_log_breadth(a, [1.0], {'p': (p,), 'both': (p,q)}, [p,q])
        self.assertAlmostEqual(result['p'], 0.0)
        self.assertAlmostEqual(result['both'], 0.0)
        a[0,0,1] = a[0,1,0] = 0
        self.assertIsNone(rule_log_breadth(a, [1.0], {'p': (p,)}, [p,q])['p'])


if __name__ == '__main__':
    unittest.main()
