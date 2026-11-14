import unittest

import numpy as np

from exposure.design_diagnostics import FEATURES, statistics as unweighted
from exposure.design_rule_allocation import integer_allocation, moments, statistics


class RuleAllocationTests(unittest.TestCase):
    def test_weighted_matches_replicated_rows(self):
        rng = np.random.default_rng(42)
        rows = []
        for rule in ('a', 'b', 'c'):
            for i in range(7):
                x = rng.normal(size=4)
                rows.append({'rule': rule, 'design': str(i), **dict(zip(FEATURES, x))})
        mean, second = moments(rows, ['a', 'b', 'c'])
        result = statistics(mean, second, [1/6, 2/6, 3/6])
        expanded = [r for r in rows for _ in range({'a':1,'b':2,'c':3}[r['rule']])]
        expected = unweighted(expanded)
        self.assertAlmostEqual(result['correlation_marginal_lift'],
                               expected['correlations']['log_mean_marginal']['log_lift'])
        for name in FEATURES:
            self.assertAlmostEqual(result['vif'][name], expected['vif'][name]['value'])

    def test_missing_or_duplicate_design_rows_rejected(self):
        row = {'rule':'a','design':'d',**dict.fromkeys(FEATURES,1)}
        with self.assertRaises(ValueError): moments([row], ['a','b'])
        with self.assertRaises(ValueError): moments([row,row], ['a'])

    def test_collinear_features_rejected(self):
        means=np.zeros((2,4)); seconds=np.tile(np.eye(4),(2,1,1)); seconds[:,0,1]=1; seconds[:,1,0]=1
        with self.assertRaises(ValueError): statistics(means,seconds,[.5,.5])

    def test_integer_allocation_preserves_total_and_deterministic_ties(self):
        np.testing.assert_array_equal(integer_allocation([.5,.25,.25], 6), [3,2,1])
        with self.assertRaises(ValueError): integer_allocation([.5,.4], 6)


if __name__ == '__main__': unittest.main()
