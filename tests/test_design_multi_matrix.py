import json
import unittest
from pathlib import Path

from exposure.design_multi_matrix import generate
from exposure.design_subspace import diagnose
from exposure.measure import estimate, load_rules


class MultiMatrixTests(unittest.TestCase):
    def test_endpoints_preserve_both_estimated_marginals(self):
        data = json.loads(Path('examples/exposure_rules.json').read_text())
        rules = load_rules(data)
        reports = [estimate(rules, samples=20, seed=seed, include_subsets=True) for seed in (11, 12)]
        for report in reports:
            report['rules'] = data
        ref = dict.fromkeys(data, 1 / len(data))
        designs = generate(reports, ref, min_weight=0)['designs']
        self.assertEqual(len(designs), 1 + 2 * len(data))
        for report in reports:
            events = diagnose(report, ref)['singleton_events']
            for design in designs:
                for event in events:
                    got = sum(design['weights'][name] * x for name, x in zip(data, event['column_probabilities'], strict=True))
                    self.assertAlmostEqual(got, event['reference_exposure'], places=6)
        with self.assertRaisesRegex(ValueError, 'seeds'):
            generate([reports[0], reports[0]], ref)


if __name__ == '__main__':
    unittest.main()
