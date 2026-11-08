# Multi-chain design diagnostics

`exposure.design_chains` runs independent hit-and-run chains from the same
reference mixture and reports convergence diagnostics on each rule's log-lift
series. A single chain cannot separate slow mixing from a narrow feasible set.

```sh
uv run python -m exposure.design_chains \
  --report outputs/exposure-57-pilot.json \
  --weights outputs/uniform-57.json \
  --chains 4 --samples 100 --seed 0 \
  --burn-in 1000 --thin 10 \
  --output outputs/design-chains.json
```

## Output

- `chain_results`: the raw designs for each chain, with its seed.
- `diagnostics.rules.<name>`: split-R-hat, minimum and median autocorrelation
  ESS, per-chain mean range, pooled range, and a `finite` flag.
- `diagnostics.summary`: rule count, max R-hat, min ESS, median and max
  per-rule range, thresholds, and a `status`.

`status` is `fail` if any R-hat exceeds `1.01` or any ESS is below `100`;
`insufficient` if a diagnostic is undefined; otherwise `pass`. These thresholds
are diagnostics, not a Stage 0 gate. Small exploratory runs are expected to
fail.

## Current pilot result

A 5-chain × 200-design run (`burn_in=1000`, `thin=10`) gave median per-rule
ranges of `0.148–0.182` and `0/57,000` positive lifts. Combined with the exact
LP bounds (`1.172` median width, `23/57` rules admitting positive lift), this
shows the sampler is not reaching the positive-lift corner. It does not prove
the chains are converged; read the reported R-hat and ESS before interpreting
the range.

## Limits

- The equality constraints use the estimated Monte Carlo matrix; uncertainty is
  not propagated.
- The sampler explores the equality nullspace only; it does not impose
  inequality constraints on lifts or pair exposures.
- The diagnostics do not establish uniformity, independence, or decorrelation.
- This is not a Stage 0 pass and does not target `beta_lambda` or
  `beta_lambdaR` directly.
