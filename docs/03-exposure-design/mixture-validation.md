# Independent mixture validation

This instrument check compares full-rule exposure predicted by the estimated
matrix with fresh scenes sampled directly from a weighted mixture of the same
conditioned renderer distributions. It evaluates scene annotations with
`satisfies`; it does not evaluate generated pixels or a trained model.

```sh
uv run python -m exposure.measure --rules examples/exposure_rules.json --samples 200 --seed 42 --include-subsets --output outputs/pilot-subsets.json
uv run python -m exposure.validate_mixture --report outputs/pilot-subsets.json --weights examples/uniform_weights.json --samples 1000 --seed 43 --output outputs/mixture-validation.json
```

The validation seed must differ from the source measurement seed. A distinct
seed avoids replaying the same pseudorandom stream, but is not a formal proof
of random independence. Reproducibility also requires the same Python and
renderer code, rule order and dependencies. Schema version 1 reports do not
record a renderer version: verify it matches before interpreting a comparison.
The output includes the source report's canonical SHA-256, rule definitions,
normalized weights, seeds, per-condition draws, and rejection attempts.

For every rule the report gives the source-matrix weighted prediction, direct
scene count and proportion, and their difference. `observed_wilson_95` is a
pointwise Wilson interval for the direct sample. The report does not quantify
uncertainty in the matrix prediction or the difference; the observed interval
alone is not a confidence interval for that difference.

The output deliberately makes no pass/fail decision. Comparisons across rules
share scenes and are correlated; the observation intervals are pointwise, not
simultaneous. If weights were estimated from the same source matrix, their
uncertainty is not accounted for. This check does not quantify renderer drift,
measurement bias, uncertainty in the weights, causality, or population-level
feasibility. Boundary events and small samples require additional care.

If any conditioned sample exceeds `--max-attempts`, the run aborts without
writing a new report; this does not prove the condition impossible. Existing
output files are left untouched when sampling fails.
