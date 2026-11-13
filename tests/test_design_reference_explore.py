import json
import unittest
from pathlib import Path

import numpy as np

from exposure.design_reference_explore import correlation, evaluate, matrix, optimize_reference


class ReferenceExploreTests(unittest.TestCase):
    def test_pairwise_vif_formula_and_constant_failure(self):
        values = np.array([[0.2, 0.4], [0.6, 0.5], [0.8, 0.9]])
        counts = np.array([0, 1, 1])
        result = evaluate(values, counts, np.array([0.5, 0.5]))
        self.assertAlmostEqual(result["pairwise_vif_lower_bound"], 1 / (1 - result["correlation"] ** 2))
        with self.assertRaises(ValueError):
            correlation(values, np.ones(3), np.array([0.5, 0.5]))

    def test_bad_bounds_rejected(self):
        with self.assertRaises(ValueError):
            optimize_reference(np.ones((3, 2)), np.array([0, 1, 2]), 0.6, 0.8)

    def test_saved_matrix_and_independent_holdout(self):
        pilot = Path("outputs/exposure-57-pilot.json")
        holdout = Path("outputs/exposure-57-holdout-81205.json")
        if not pilot.exists() or not holdout.exists():
            self.skipTest("Saved exposure reports are unavailable")
        names, train, counts = matrix(json.loads(pilot.read_text()))
        held_names, held, held_counts = matrix(json.loads(holdout.read_text()))
        self.assertEqual(names, held_names)
        np.testing.assert_array_equal(counts, held_counts)
        weights = optimize_reference(train, counts)
        self.assertAlmostEqual(float(weights.sum()), 1)
        self.assertGreaterEqual(float(weights.min()), 0.001 - 1e-7)
        self.assertLessEqual(float(weights.max()), 0.04 + 1e-7)
        uniform = np.full(len(names), 1 / len(names))
        self.assertLess(evaluate(held, counts, weights)["pairwise_vif_lower_bound"],
                        evaluate(held, counts, uniform)["pairwise_vif_lower_bound"])


if __name__ == "__main__":
    unittest.main()
