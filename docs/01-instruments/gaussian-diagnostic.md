# Gaussian-mixture denoiser diagnostic

This CPU diagnostic checks noise-prediction loss calculations against a known,
one-dimensional Bayes denoiser. It is a numerical instrument check; it does not
train or evaluate the proposed neural network and is not a Stage 0 gate.

```sh
uv run python -m exposure.gaussian_diagnostic --means -2 2 --weights 0.5 0.5 --variance 1 --alpha 0.5 --samples 100000 --seed 0 --output outputs/gaussian-diagnostic.json
```

The clean distribution is a mixture of Gaussians with the specified means,
positive normalized weights, and a common positive variance `variance`. Given
`alpha` strictly between zero and one, the noising process is
`x_t = sqrt(alpha) * x_0 + sqrt(1 - alpha) * epsilon`, with independent standard
normal noise. Thus `alpha` is cumulative signal power, not a timestep index.

The noisy distribution for each component is Gaussian with mean
`sqrt(alpha) * mean` and variance `v = alpha * variance + 1 - alpha`. Posterior
component probabilities are evaluated with log-sum-exp. Within each component,
the conditional noise mean is `sqrt(1 - alpha) * (x_t - sqrt(alpha) * mean) / v`
and the conditional variance is `alpha * variance / v`. The Bayes prediction is
the posterior-weighted average of the component conditional means.

The reference `quadrature_mse` integrates the conditional noise variance,
including variance between the posterior component means, against the noisy
mixture distribution. Quadrature is carried out separately per component to
avoid missing separated modes. `quadrature_error_estimate` is the weighted
numerical integration error estimate, not a statistical confidence interval.
The Monte Carlo estimate samples clean values and independent noise, evaluates
the squared residual, and reports its standard error. The samples and seed are
recorded for reproduction. Very small Monte Carlo standard errors do not measure
all numerical error, distribution mismatch, or uncertainty about a trained model.
