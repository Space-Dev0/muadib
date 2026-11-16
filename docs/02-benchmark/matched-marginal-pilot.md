# Matched-marginal pilot

These examples use the prototype measurement matrix generated with 200
accepted scenes per condition, seed 42, and `examples/exposure_rules.json`.
They hold only the **red** and **square** singleton scene exposures at the
uniform-mixture values 0.77625 and 0.81375. They target a full
`red_left_of_square` rule exposure no greater than 0.32 or no less than 0.55.

```sh
uv run python -m exposure.measure --rules examples/exposure_rules.json --samples 200 --seed 42 --include-subsets --output outputs/pilot-subsets.json
uv run python -m exposure.design --report outputs/pilot-subsets.json --constraints examples/fixed_marginals_low.json --objective min-kl --output outputs/pilot-low.json
uv run python -m exposure.design --report outputs/pilot-subsets.json --constraints examples/fixed_marginals_high.json --objective min-kl --output outputs/pilot-high.json
```

On this specific pilot matrix, both designs are feasible. The target values
were selected using this matrix, so a different seed, sample count, renderer,
or rule set may make them infeasible. Compare the outputs' `constraints`,
`weights`, `max_violation`, and `kl_to_uniform` fields.

These are illustrative **scene-level** estimates. The relation marginal and
other predicate and pair exposures are not held fixed. Therefore this example
does not demonstrate a change in lift at fixed full predicate marginals, a
causal effect, or a result from a trained model. Monte Carlo uncertainty is not
propagated. Confirmatory designs require all relevant marginals to be controlled
and the proposal's feasibility and decorrelation gates to be met.
