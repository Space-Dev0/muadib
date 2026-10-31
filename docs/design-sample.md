# Exploratory marginal-preserving design sampling

`exposure.design_sample` performs seeded hit-and-run steps in the nullspace of
the estimated singleton exposure matrix and simplex equality. Its output can be
used as the `designs` input of `exposure.design_diagnostics`:

```sh
uv run python -m exposure.design_sample --report outputs/exposure-subsets.json --weights examples/uniform_weights.json --samples 20 --seed 42 --min-weight 0.01 --burn-in 100 --thin 10 --output outputs/design-samples.json
```

The input report must have `--include-subsets`; weights must name every
condition and sum to one within `1e-12`. The weights are never renormalized.
The reference must be strictly above the requested floor, and the equality
nullspace must have nonzero dimension. The existing four-condition pilot can
have nullity zero; in that case sampling fails clearly instead of fabricating
different designs. A larger, suitable rule set or a different experimental
design is needed.

At each step, a random nullspace direction is selected, and the complete
feasible interval along that direction is calculated from the simplex floor.
The algorithm checks simplex and estimated singleton equality residuals to
`1e-7`, does not clip or renormalize weights, and errors on a degenerate step or
boundary. Output contains only `designs`, each with `name` and `weights`.

The sampler does **not** impose additional rule-exposure or other inequality
constraints. Samples are correlated, and burn-in and thinning do not prove
mixing, independence, uniformity, or convergence. The estimated matrix has
Monte Carlo uncertainty; sampled directions do not certify decorrelation,
VIF criteria, or a Stage 0 pass.
