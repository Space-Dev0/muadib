import json
import math
import subprocess
import sys
import unittest

from exposure.composition import (
    composition_gaps,
    gamma_log,
    gamma_logit,
    logit,
    null_gaps,
)


class CompositionGapTests(unittest.TestCase):
    def test_logit_round_trip_and_boundaries(self):
        for probability in (1e-6, 0.05, 0.5, 0.95, 1 - 1e-6):
            self.assertAlmostEqual(1 / (1 + math.exp(-logit(probability))), probability)
        for probability in (0.0, 1.0, -0.1, 1.1, float("nan"), True):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                logit(probability)

    def test_stated_gamma_is_zero_under_multiplicative_null(self):
        for marginals in ((0.8, 0.8), (0.8, 0.8, 0.8), (0.9, 0.9, 0.9),
                          (0.58, 0.72, 0.999), (0.2, 0.3, 0.4, 0.5)):
            gaps = null_gaps(marginals)
            self.assertAlmostEqual(gaps["rsr"], math.prod(marginals))
            self.assertAlmostEqual(gaps["gamma_logit"], 0.0, places=12)
            self.assertAlmostEqual(gaps["gamma_log"], 0.0, places=12)
            self.assertEqual(gaps["gamma_status"], "both_defined")

    def test_review_expression_is_a_different_estimand(self):
        # The review computed logit(prod c_p) - sum logit(c_p).  That is not
        # the proposal's Gamma and is strongly negative even under the null.
        marginals = (0.8, 0.8)
        product = math.prod(marginals)
        wrong = logit(product) - sum(logit(value) for value in marginals)
        self.assertAlmostEqual(wrong, -2.1972245773362196)
        self.assertAlmostEqual(gamma_logit(product, marginals), 0.0, places=12)

    def test_log_scale_is_additive_and_defined_at_one(self):
        marginals = (0.5, 1.0)
        self.assertEqual(composition_gaps(0.5, marginals)["gamma_logit"], 0.0)
        self.assertEqual(gamma_logit(0.5, marginals), 0.0)
        self.assertAlmostEqual(gamma_log(1.0, marginals), math.log(2.0))
        gaps = composition_gaps(1.0, marginals)
        self.assertIsNone(gaps["gamma_logit"])
        self.assertAlmostEqual(gaps["gamma_log"], math.log(2.0))
        self.assertEqual(gaps["gamma_status"], "log_only")

    def test_zero_and_invalid_inputs_are_explicit(self):
        gaps = composition_gaps(0.0, (0.5, 0.5))
        self.assertIsNone(gaps["gamma_logit"])
        self.assertIsNone(gaps["gamma_log"])
        self.assertEqual(gaps["gamma_status"], "undefined")
        zero_marginal = composition_gaps(0.5, (0.0,))
        self.assertIsNone(zero_marginal["gamma_logit"])
        self.assertIsNone(zero_marginal["gamma_log"])
        self.assertEqual(zero_marginal["gamma_status"], "undefined")
        for rsr, marginals in ((0.5, ()), (0.5, (1.5,)),
                               (-0.1, (0.5,)), (1.1, (0.5,)), (float("nan"), (0.5,))):
            with self.subTest(rsr=rsr, marginals=marginals), self.assertRaises(ValueError):
                composition_gaps(rsr, marginals)

    def test_cli_null(self):
        result = subprocess.run(
            [sys.executable, "-m", "exposure.composition", "--rsr", "0.5",
             "--marginals", "0.8", "0.8", "--null"],
            check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        self.assertAlmostEqual(data["gamma_logit"], 0.0, places=12)
        self.assertAlmostEqual(data["gamma_log"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
