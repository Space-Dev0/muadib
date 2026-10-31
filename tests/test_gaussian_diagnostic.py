import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exposure.gaussian_diagnostic import GaussianMixture, expected_noise_mse, predict_noise, run_diagnostic


class GaussianDiagnosticTests(unittest.TestCase):
    def test_single_gaussian_closed_form(self):
        mixture = GaussianMixture((1.25,), (1.0,), 2.5)
        for alpha in (0.1, 0.5, 0.9):
            reference, error = expected_noise_mse(mixture, alpha)
            expected = alpha * mixture.variance / (alpha * mixture.variance + 1 - alpha)
            self.assertAlmostEqual(reference, expected, places=9)
            self.assertLess(error, 1e-8)
            x = 2.0
            expected_prediction = math.sqrt(1 - alpha) * (x - math.sqrt(alpha) * 1.25) / (
                alpha * mixture.variance + 1 - alpha)
            self.assertAlmostEqual(float(predict_noise(mixture, alpha, x)), expected_prediction)

    def test_mixture_quadrature_matches_independent_monte_carlo(self):
        mixture = GaussianMixture((-2.0, 1.0, 3.0), (0.2, 0.5, 0.3), 0.35)
        report = run_diagnostic(mixture, 0.6, samples=250_000, seed=12)
        # Five Monte Carlo standard errors is a conservative stochastic tolerance.
        self.assertLess(abs(report["monte_carlo_mse"] - report["quadrature_mse"]),
                        5 * report["monte_carlo_standard_error"])
        self.assertGreater(report["quadrature_mse"], 0)
        self.assertLess(report["quadrature_mse"], 1)

    def test_symmetric_denoiser_and_stable_weights(self):
        mixture = GaussianMixture((-3.0, 3.0), (0.5, 0.5), 1.0)
        xs = np.array([-100.0, -3.0, 0.0, 3.0, 100.0])
        estimates = predict_noise(mixture, 0.5, xs)
        np.testing.assert_allclose(estimates, -estimates[::-1], atol=1e-12)
        self.assertTrue(np.all(np.isfinite(estimates)))

    def test_translation_preserves_predictions_and_risk(self):
        base = GaussianMixture((-1.0, 2.0), (0.3, 0.7), 0.4)
        shifted = GaussianMixture((4.0, 7.0), (0.3, 0.7), 0.4)
        alpha = 0.7
        xs = np.array([-1.0, 0.0, 2.0])
        np.testing.assert_allclose(predict_noise(base, alpha, xs),
                                   predict_noise(shifted, alpha, xs + 5 * math.sqrt(alpha)), atol=1e-12)
        self.assertAlmostEqual(expected_noise_mse(base, alpha)[0], expected_noise_mse(shifted, alpha)[0], places=10)

    def test_invalid_configuration(self):
        for means, weights, variance in (((), (), 1), ((0,), (0,), 1), ((0, 1), (1,), 1),
                                          ((0, 1), (0.2, 0.2), 1), ((0,), (1,), 0)):
            with self.subTest(means=means, weights=weights), self.assertRaises(ValueError):
                GaussianMixture(means, weights, variance)
        mixture = GaussianMixture((0,), (1,), 1)
        for alpha in (0, 1, -1, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                expected_noise_mse(mixture, alpha)
        with self.assertRaises(ValueError):
            run_diagnostic(mixture, 0.5, samples=1, seed=0)

    def test_cli_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            cmd = [sys.executable, "-m", "exposure.gaussian_diagnostic", "--samples", "500",
                   "--seed", "7", "--output", str(output)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            first = json.loads(output.read_text())
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertEqual(first, json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
