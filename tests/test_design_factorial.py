import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from exposure.design_factorial import features, score_gradient, contrasts, optimize

class FactorialDesignTests(unittest.TestCase):
    def setUp(self):
        self.marg=np.array([[[.2,.4,.7],[.3,.6,.8]],[[.3,.7,.6],[.2,.4,.7]]])
        self.joint=np.array([[.1,.25,.55],[.08,.3,.4]])
        self.w=np.array([[.2,.3,.5],[.5,.2,.3],[.3,.5,.2],[.4,.3,.3]])

    def test_gradient_matches_finite_difference(self):
        flat=self.w.ravel()
        analytic=score_gradient(flat,self.marg,self.joint)[1]
        numeric=approx_derivative(lambda z: score_gradient(z,self.marg,self.joint)[0],flat,method='3-point').ravel()
        np.testing.assert_allclose(analytic,numeric,rtol=2e-5,atol=2e-5)
        expected=contrasts(flat,self.marg,self.joint,.01,.02,.05)[1]
        actual=approx_derivative(lambda z: contrasts(z,self.marg,self.joint,.01,.02,.05)[0],flat,method='3-point')
        np.testing.assert_allclose(expected,actual,rtol=2e-5,atol=2e-5)

    def test_constant_features_cannot_satisfy_positive_contrast(self):
        m=np.full((1,2,3),.5); n=np.full((1,3),.25)
        result=optimize(m,n,starts=1,maxiter=5)
        self.assertIsNone(result['best'])
        self.assertFalse(result['candidates'][0]['feasible'])

    def test_impossible_weight_cap_and_mismatched_extra_rejected(self):
        with self.assertRaises(ValueError):
            optimize(self.marg, self.joint, max_weight=.2, starts=1)
        with self.assertRaises(ValueError):
            optimize(self.marg, self.joint, extra_matrices=[(self.marg, self.joint[:1])], starts=1)

    def test_second_matrix_cannot_be_ignored(self):
        constant_m = np.full_like(self.marg, .5)
        constant_j = np.full_like(self.joint, .25)
        result = optimize(self.marg, self.joint, extra_matrices=[(constant_m, constant_j)],
                          starts=1, maxiter=10)
        self.assertIsNone(result['best'])
        self.assertEqual(result['settings']['fitting_matrices'], 2)

    def test_zero_exposure_rejected(self):
        with self.assertRaises(ValueError): features(self.w,np.zeros_like(self.marg),self.joint)

if __name__ == '__main__': unittest.main()
