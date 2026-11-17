import json
import unittest
import tempfile
from unittest import mock
from pathlib import Path
from exposure.measure import estimate, load_rules
from exposure.measure_fast import estimate as fast_estimate


class FastMeasureTests(unittest.TestCase):
    def test_exactly_reproduces_original(self):
        data = json.loads(Path('examples/exposure_rules.json').read_text())
        rules = load_rules(data)
        for seed in (3, 41):
            with self.subTest(seed=seed):
                expected = estimate(rules, samples=30, seed=seed, include_subsets=True)
                self.assertEqual(fast_estimate(rules, samples=30, seed=seed), expected)

    def test_rejects_unhandled_variable_names(self):
        from exposure.rules import Attribute
        with self.assertRaisesRegex(ValueError, 'a/b/c'):
            fast_estimate({'bad': (Attribute('d', 'color', 'red'),)}, samples=1, seed=0)

    def test_interrupted_run_resumes_exactly_and_preserves_breadth(self):
        from exposure import measure_fast
        from exposure.design_breadth import estimate as breadth_estimate
        data = json.loads(Path('examples/exposure_rules.json').read_text())
        rules = load_rules(data)
        original = measure_fast.sample_conditioned
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 3:
                raise RuntimeError('simulated interruption')
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'state.json'
            with mock.patch.object(measure_fast, 'sample_conditioned', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    fast_estimate(rules, samples=3, seed=23, state_path=state)
            resumed, breadth = fast_estimate(rules, samples=3, seed=23, state_path=state, collect_breadth=True)
            expected = estimate(rules, samples=3, seed=23, include_subsets=True)
            self.assertEqual(resumed, expected)
            expected['rules'] = data
            self.assertEqual(breadth['counts_by_condition'], breadth_estimate(expected)['counts_by_condition'])
            with self.assertRaisesRegex(ValueError, 'does not match'):
                fast_estimate(rules, samples=3, seed=24, state_path=state)


if __name__ == '__main__':
    unittest.main()
