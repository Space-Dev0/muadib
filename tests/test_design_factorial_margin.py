import unittest

import numpy as np

from exposure.design_factorial import contrasts
from exposure.design_factorial_margin import optimize


class FactorialMarginTests(unittest.TestCase):
    def test_fit_respects_constraints_and_improves_minimum_margin(self):
        initial = np.full((4, 3), 1 / 3)
        marginal = np.array([[[.2, .6, .8], [.3, .7, .5]],
                             [[.4, .7, .3], [.6, .2, .8]]])
        joint = np.array([[.1, .4, .3], [.2, .3, .15]])
        before = contrasts(initial.ravel(), marginal, joint, .02, .02, .05)[0].min()
        fit = optimize(initial, [(marginal, joint)], min_weight=0, max_weight=1)
        self.assertTrue(fit['feasible'], fit['message'])
        fitted = np.asarray(fit['weights'])
        self.assertTrue(np.allclose(fitted.sum(axis=1), 1))
        actual = contrasts(fitted.ravel(), marginal, joint, .02, .02, .05)[0].min()
        self.assertGreaterEqual(actual, before - 1e-7)
        self.assertAlmostEqual(actual, fit['margin'], places=6)
        self.assertEqual(len(fit['information_eigenvalues'][0]), 2)

    def test_bad_weights_and_matrix_rejected(self):
        marginal = np.ones((1, 2, 3)) * .5
        joint = np.ones((1, 3)) * .2
        with self.assertRaises(ValueError):
            optimize(np.ones((4, 3)), [(marginal, joint)], min_weight=0, max_weight=1)
        with self.assertRaises(ValueError):
            optimize(np.full((4, 3), 1 / 3), [(marginal, np.ones((1, 2)))],
                     min_weight=0, max_weight=1)

    def test_solver_success_with_negative_margin_is_not_feasible(self):
        initial = np.full((4, 3), 1 / 3)
        marginal = np.ones((1, 2, 3)) * .5
        joint = np.ones((1, 3)) * .2
        fit = optimize(initial, [(marginal, joint)], min_weight=0, max_weight=1)
        self.assertTrue(fit['solver_success'])
        self.assertLess(fit['margin'], 0)
        self.assertFalse(fit['feasible'])


if __name__ == '__main__':
    unittest.main()
