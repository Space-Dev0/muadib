# Exploratory design diagnostics

`exposure.design_diagnostics` pools estimated rule-by-design rows and reports
Pearson correlations and variance inflation factors (VIF) for log mean
predicate marginal, rule log lift, attribute-predicate count, and relation
count. It does not create exposure designs.

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
