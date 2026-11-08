# Exploratory design diagnostics

`exposure.design_diagnostics` pools estimated rule-by-design rows and reports
Pearson correlations and variance inflation factors (VIF) for log mean
predicate marginal, rule log lift, attribute-predicate count, and relation
count. It does not create exposure designs. Use the
[design sampler](design-sample.md) to generate exploratory designs that
preserve estimated singleton exposures.

Prepare a subset-enabled measurement report and a separate JSON file whose
`designs` list contains at least three distinct names and complete weight maps:

```json
{"designs": [
  {"name": "uniform", "weights": {"red": 0.25, "square": 0.25, "red_left_of_square": 0.25, "blue_above_circle": 0.25}},
  {"name": "red-heavy", "weights": {"red": 0.7, "square": 0.1, "red_left_of_square": 0.1, "blue_above_circle": 0.1}},
  {"name": "square-heavy", "weights": {"red": 0.1, "square": 0.7, "red_left_of_square": 0.1, "blue_above_circle": 0.1}}
]}
```

For example, save that JSON as `outputs/candidate-designs.json`, then run:

```sh
uv run python -m exposure.design_diagnostics --report outputs/exposure-subsets.json --designs outputs/candidate-designs.json --output outputs/design-diagnostics.json
```

Every weight map must match the report's conditions and satisfy the mixture
validator. Each output row identifies its design and rule. Rows with a zero
mean marginal or undefined log lift are excluded with explicit reasons; no
small constant is substituted. Correlations are null for constant features or
too few observations. VIF is null with a status when there are too few rows,
the target is constant, or it is exactly or nearly explained by the others.

`examples/pilot_designs.json` is a reproducible alternative containing the
uniform, low, and high allocations from the matched-marginal pilot. It is tied
to that pilot's 200-scene, seed-42 rule matrix; see
[matched-marginal pilot](matched-marginal-pilot.md). For example:

```sh
uv run python -m exposure.design_diagnostics --report outputs/exposure-subsets.json --designs examples/pilot_designs.json --output outputs/design-diagnostics.json
```

These are **exploratory** summaries, not a Stage 0 go/no-go decision. Rows
share rules and designs, exposure columns are estimated from finite
Monte Carlo samples, uncertainty is not propagated, and breadth is omitted.
Pooled correlations and VIF do not themselves establish causal effects or
confirm that the proposed decorrelation criteria are met.

Two level issues are now verified:

- **Pooled correlations are between-rule.** Within a rule the design fixes
  singleton marginals exactly, so `log m̄` has zero within-rule variation
  (observed range `2.2e−16`). The pooled `r` reflects rule-level differences in
  mean lift, not a within-rule failure. `exposure.design_strata` reports both
  levels.
- **`β_m` is not identified.** Because `log m̄` is constant within rule, `β_m`
  is exactly collinear with the rule intercept. Drop it or vary marginals within
  rule before claiming a decorrelation gate.

By the proposal's literal criterion (`|r| ≤ 0.3`, `VIF < 2`), the current pilot
fails: pooled `r = −0.446`, VIF `11.61` for log lift and `13.72` for relation
count. The gate must be redefined on an identified within-rule quantity, or
Stage 0 is not passed. Exact LP bounds and multi-chain diagnostics are
available in [design bounds](design-bounds.md) and
[design chains](design-chains.md).
