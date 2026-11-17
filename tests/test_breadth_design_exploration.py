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


if __name__ == '__main__':
    unittest.main()
