import unittest
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from exposure.design_factorial import contrasts
from exposure.design_factorial_uncertainty import interval_audit


class FactorialUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([[.8, .2], [.6, .4], [.4, .6], [.2, .8]])
        self.marg = np.array([[[.2, .8], [.3, .7]]])
        self.joint = np.array([[.1, .6]])

    def audit(self, n, **overrides):
        settings = {'delta_m': .02, 'delta_l': .02, 'cross_tolerance': .05}
        settings.update(overrides)
        return interval_audit(self.weights, self.marg, self.joint, n, **settings)

    def test_intervals_contain_point_estimates_and_match_constraints(self):
        result = self.audit([10000, 5000])
        self.assertEqual(result['family_size'], 12)
        self.assertEqual(len(result['constraints']), 12)
        for row in result['constraints']:
            if row['lower'] is not None:
                self.assertLessEqual(row['lower'], row['estimate'])
            if row['upper'] is not None:
                self.assertGreaterEqual(row['upper'], row['estimate'])
        self.assertEqual(sum(result['counts'].values()), 12)

    def test_small_sample_is_uncertain_not_silently_clipped(self):
        result = self.audit([1, 1])
        self.assertEqual(result['counts']['unresolved'], 12)
        self.assertTrue(any(row['lower'] is None for row in result['constraints']))

    def test_large_sample_can_certify_violations(self):
        result = self.audit([10**12, 10**12], delta_m=1.0)
        self.assertEqual(result['constraints'][0]['status'], 'certified_violated')
        self.assertLess(result['constraints'][0]['upper'], 0)

    def test_propagated_bounds_cover_perturbed_population_margins(self):
        result = self.audit([1000, 1000])
        population_marg = self.marg + np.array([[[.01, -.01], [-.01, .01]]])
        population_joint = self.joint + np.array([[.01, -.01]])
        actual = contrasts(self.weights.ravel(), population_marg, population_joint,
                           .02, .02, .05)[0]
        for margin, row in zip(actual, result['constraints'], strict=True):
            if row['lower'] is not None:
                self.assertLessEqual(row['lower'], margin)
            if row['upper'] is not None:
                self.assertGreaterEqual(row['upper'], margin)

    def test_invalid_inputs_fail_closed(self):
        for samples in ([0, 1], [1.5, 2], [1]):
            with self.assertRaises(ValueError):
                self.audit(samples)
        with self.assertRaises(ValueError):
            self.audit([10, 10], alpha=0)
        with self.assertRaises(ValueError):
            interval_audit(self.weights * 2, self.marg, self.joint, [10, 10],
                           delta_m=.02, delta_l=.02, cross_tolerance=.05)

    def test_cli_rejects_reused_seed_and_tampered_counts(self):
        def report(seed):
            return {'seed': seed, 'rule_order': ['c0', 'c1'],
                    'estimator': 'fake', 'estimator_sha256': 'fake-hash',
                    'rules': {'r': [{}, {}]}, 'columns': [
                        {'condition': name, 'accepted_scenes': 100,
                         'measurements': {'r': {'successes': 20, 'probability': .2}},
                         'predicate_subsets': {'r': [
                             {'predicate_indices': [0], 'successes': 50, 'probability': .5},
                             {'predicate_indices': [1], 'successes': 50, 'probability': .5}]}}
                        for name in ('c0', 'c1')]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.json'
            reference.write_text(json.dumps(report(1)))
            design = {'best': {'feasible': True, 'weights': self.weights.tolist()},
                      'settings': {'delta_m': .02, 'delta_l': .02, 'cross_tolerance': .05},
                      'report_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                      'fitting_seeds': [1], 'rule_names': ['r'], 'condition_order': ['c0', 'c1']}
            (root / 'design.json').write_text(json.dumps(design))
            holdout = root / 'holdout.json'
            def run(data):
                holdout.write_text(json.dumps(data))
                return subprocess.run([sys.executable, '-m', 'exposure.design_factorial_uncertainty',
                                       '--design', str(root / 'design.json'), '--reference-report', str(reference),
                                       '--report', str(holdout), '--output', str(root / 'out.json')],
                                      capture_output=True, text=True)
            self.assertEqual(run(report(2)).returncode, 0)
            result = json.loads((root / 'out.json').read_text())
            self.assertEqual(result['evaluated_thresholds'],
                             {'delta_m': .02, 'delta_l': .02, 'cross_tolerance': .05})
            overridden = subprocess.run(
                [sys.executable, '-m', 'exposure.design_factorial_uncertainty',
                 '--design', str(root / 'design.json'), '--reference-report', str(reference),
                 '--report', str(holdout), '--output', str(root / 'override.json'),
                 '--delta-m', '.03', '--cross-tolerance', '.04'], capture_output=True, text=True)
            self.assertEqual(overridden.returncode, 0, overridden.stderr)
            self.assertEqual(json.loads((root / 'override.json').read_text())['evaluated_thresholds'],
                             {'delta_m': .03, 'delta_l': .02, 'cross_tolerance': .04})
            self.assertNotEqual(run(report(1)).returncode, 0)
            tampered = report(3)
            tampered['columns'][0]['measurements']['r']['successes'] = 99
            self.assertNotEqual(run(tampered).returncode, 0)


if __name__ == '__main__':
    unittest.main()
