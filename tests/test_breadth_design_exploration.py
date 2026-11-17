import importlib.util
import itertools
import unittest
from pathlib import Path


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


if __name__ == '__main__':
    unittest.main()
