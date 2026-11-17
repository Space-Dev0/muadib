import unittest
import numpy as np
from exposure.design_optimal import solve


class DOptimalTests(unittest.TestCase):
    def test_linear_regression_endpoints(self):
        result = solve([[1,-1],[1,0],[1,1]])
        np.testing.assert_allclose(result['weights'], [.5, 0, .5], atol=1e-5)
        self.assertAlmostEqual(result['log_determinant'], 0., places=6)
        self.assertLess(result['max_constraint_violation'], 1e-7)

    def test_respects_linear_exposure_bound(self):
        result = solve([[1,-1],[1,0],[1,1]], constraints=[{'coefficients':[1,0,0], 'max':.2}])
        self.assertLessEqual(result['weights'][0], .2000001)
        self.assertAlmostEqual(sum(result['weights']), 1.)
        self.assertTrue(all(value > 0 for value in result['information_eigenvalues']))

    def test_rejects_unidentifiable_or_infeasible(self):
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            solve([[1,1],[1,1]])
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            solve([[1,-1],[1,1]], constraints=[{'coefficients':[1,0], 'min':.9}, {'coefficients':[0,1], 'min':.9}])

if __name__ == '__main__': unittest.main()
