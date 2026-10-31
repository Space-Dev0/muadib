"""Analytic one-dimensional diffusion noise-prediction diagnostic.

The quadrature calculation and Monte Carlo simulation both use the exact
Bayes denoiser. They do not measure a neural network or certify a benchmark.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp


@dataclass(frozen=True)
class GaussianMixture:
    means: tuple[float, ...]
    weights: tuple[float, ...]
    variance: float

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if means.ndim != 1 or means.size == 0 or not np.all(np.isfinite(means)):
            raise ValueError("means must be a nonempty finite 1D sequence")
        if (weights.shape != means.shape or not np.all(np.isfinite(weights)) or
                np.any(weights <= 0) or not math.isclose(float(weights.sum()), 1.0, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("weights must be positive, finite, match means, and sum to 1")
        if isinstance(self.variance, bool) or not math.isfinite(self.variance) or self.variance <= 0:
            raise ValueError("variance must be finite and positive")
        object.__setattr__(self, "means", tuple(float(x) for x in means))
        object.__setattr__(self, "weights", tuple(float(x) for x in weights / weights.sum()))


def _alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    return float(alpha)


def _posterior(mixture: GaussianMixture, alpha: float, x_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return component responsibilities and component-conditional noise means."""
    variance = alpha * mixture.variance + (1 - alpha)
    means_t = math.sqrt(alpha) * np.asarray(mixture.means)
    residual = np.expand_dims(x_t, -1) - means_t
    logits = np.log(mixture.weights) - 0.5 * residual**2 / variance
    responsibilities = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    conditional_noise = math.sqrt(1 - alpha) * residual / variance
    return responsibilities, conditional_noise


def predict_noise(mixture: GaussianMixture, alpha: float, x_t: float | np.ndarray) -> np.ndarray:
    """Return E[epsilon | X_t] for one or more noisy observations."""
    alpha = _alpha(alpha)
    x_t = np.asarray(x_t, dtype=float)
    if not np.all(np.isfinite(x_t)):
        raise ValueError("x_t must be finite")
    responsibility, noise_means = _posterior(mixture, alpha, x_t)
    return np.sum(responsibility * noise_means, axis=-1)


def expected_noise_mse(mixture: GaussianMixture, alpha: float) -> tuple[float, float]:
    """Integrate E[Var(epsilon | X_t)] by component-wise Gaussian quadrature.

    Returns the expected squared noise error and QUADPACK's weighted absolute
    error estimate. The latter is numerical integration error, not a confidence
    interval or a bound on floating-point error.
    """
    alpha = _alpha(alpha)
    obs_var = alpha * mixture.variance + (1 - alpha)
    within = alpha * mixture.variance / obs_var
    obs_std = math.sqrt(obs_var)
    normalizer = math.sqrt(2 * math.pi)

    def integrand(z: float, mean: float) -> float:
        x_t = np.asarray(math.sqrt(alpha) * mean + obs_std * z)
        responsibilities, noise_means = _posterior(mixture, alpha, x_t)
        average = np.sum(responsibilities * noise_means)
        between = np.sum(responsibilities * (noise_means - average) ** 2)
        return float((within + between) * math.exp(-z * z / 2) / normalizer)

    estimates = [quad(integrand, -np.inf, np.inf, args=(mean,), epsabs=1e-10, epsrel=1e-10)
                 for mean in mixture.means]
    return (math.fsum(w * result[0] for w, result in zip(mixture.weights, estimates, strict=True)),
            math.fsum(w * result[1] for w, result in zip(mixture.weights, estimates, strict=True)))


def run_diagnostic(mixture: GaussianMixture, alpha: float, *, samples: int, seed: int) -> dict:
    """Compare the quadrature reference with seeded Monte Carlo squared error."""
    alpha = _alpha(alpha)
    if type(samples) is not int or samples < 2:
        raise ValueError("samples must be an integer of at least 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    component = rng.choice(len(mixture.means), size=samples, p=mixture.weights)
    clean = np.asarray(mixture.means)[component] + math.sqrt(mixture.variance) * rng.standard_normal(samples)
    noise = rng.standard_normal(samples)
    x_t = math.sqrt(alpha) * clean + math.sqrt(1 - alpha) * noise
    errors = (predict_noise(mixture, alpha, x_t) - noise) ** 2
    reference, numerical_error = expected_noise_mse(mixture, alpha)
    return {"means": list(mixture.means), "weights": list(mixture.weights),
            "component_variance": mixture.variance, "alpha": alpha, "samples": samples, "seed": seed,
            "quadrature_mse": reference, "quadrature_error_estimate": numerical_error,
            "monte_carlo_mse": float(np.mean(errors)),
            "monte_carlo_standard_error": float(np.std(errors, ddof=1) / math.sqrt(samples)),
            "note": "1D Bayes denoiser diagnostic; not a trained model or Stage 0 gate."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check analytic Gaussian-mixture denoising loss")
    parser.add_argument("--means", nargs="+", type=float, default=[-2.0, 2.0])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.5, 0.5])
    parser.add_argument("--variance", type=float, default=1.0, help="common component variance")
    parser.add_argument("--alpha", type=float, default=0.5, help="cumulative diffusion signal power")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/gaussian-diagnostic.json"))
    args = parser.parse_args()
    try:
        result = run_diagnostic(GaussianMixture(tuple(args.means), tuple(args.weights), args.variance),
                                args.alpha, samples=args.samples, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
