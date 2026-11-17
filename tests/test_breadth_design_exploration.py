import importlib.util
import itertools
import unittest
from pathlib import Path
import numpy as np


class BreadthDesignTests(unittest.TestCase):
    def test_orthogonal_features_have_unit_vif(self):
        spec = importlib.util.spec_from_file_location('breadth_explore',
                    Path('scripts/explore-breadth-design.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        fields = (*module.FEATURES, 'log_breadth')
        rows = [dict(zip(fields, values, strict=True)) for values in itertools.product((-1., 1.), repeat=5)]
        result = module.full_statistics(rows, [1.] * len(rows))
        self.assertAlmostEqual(result['marginal_lift_r'], 0.)
        for value in result['vif'].values():
            self.assertAlmostEqual(value, 1.)
        grouped = [{**row, 'rule': f'r{index % 2}'} for index, row in enumerate(rows)]
        means, seconds = module.breadth_moments(grouped, ['r0', 'r1'])
        weighted = module.weighted_moment_statistics(means, seconds, [0.5, 0.5])
        self.assertAlmostEqual(weighted['marginal_lift_r'], result['marginal_lift_r'])
        for feature, value in result['vif'].items():
            self.assertAlmostEqual(weighted['vif'][feature], value)

    def test_reference_indicator_is_included(self):
        spec = importlib.util.spec_from_file_location('breadth_check',
                    Path('scripts/check-frozen-breadth-design.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        features = ('log_mean_marginal', 'log_lift', 'k_attr', 'k_rel')
        fields = (*features, 'log_breadth')
        rows = [dict(zip(fields, values, strict=True)) | {'design': group + '_x'}
                for values in itertools.product((-1., 1.), repeat=5) for group in ('uniform', 'fitted')]
        result = module.reference_adjusted_vifs(rows, [1.] * len(rows), features)
        self.assertIn('reference_group', result)
        for value in result.values():
            self.assertAlmostEqual(value, 1.)

    def test_group_specific_moments_have_expected_vifs(self):
        spec = importlib.util.spec_from_file_location('ref_alloc',
                    Path('scripts/explore-reference-allocation.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        fields = module.FEATURE_NAMES[:-1]
        rows = [dict(zip(fields, values, strict=True)) |
                {'design': group + '_x', 'rule': rule, 'reference_group': float(group == 'fitted')}
                for values in itertools.product((-1., 1.), repeat=5)
                for group in ('uniform', 'fitted') for rule in ('a', 'b')]
        means, seconds = module.moments(rows, ['a', 'b'])
        result = module.statistics(means, seconds, [0.25] * 4)
        self.assertAlmostEqual(result['marginal_lift_r'], 0.)
        for value in result['vif'].values():
            self.assertAlmostEqual(value, 1.)
        equality, target = module.equal_rule_constraints(2)
        np.testing.assert_allclose(equality @ np.array([.4, .1, .1, .4]), target)
        self.assertGreater(np.max(np.abs(equality @ np.array([.4, .1, .2, .3]) - target)), 0)

    def test_design_weight_gradient_matches_finite_differences(self):
        spec = importlib.util.spec_from_file_location('equal_designs', Path('scripts/explore-equal-rule-designs.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        rng = np.random.default_rng(14)
        arrays = [rng.normal(size=(80, 6)) + rng.normal(size=6) for _ in range(8)]
        means = np.stack([a.mean(axis=0) for a in arrays])
        seconds = np.stack([a.T @ a / len(a) for a in arrays])
        weights = np.full(8, 1/8); direction = np.zeros(8); direction[0] = 1; direction[-1] = -1
        value, gradient = module.objective_gradient(weights, [(means, seconds)])
        eps = 1e-6
        plus, _ = module.objective_gradient(weights + eps*direction, [(means, seconds)])
        minus, _ = module.objective_gradient(weights - eps*direction, [(means, seconds)])
        self.assertAlmostEqual(float(gradient @ direction), (plus-minus)/(2*eps), places=5)


if __name__ == '__main__':
    unittest.main()
