# Within-rule exposure identifiability diagnostic

`exposure.design_identifiability` examines the rank of three predictor columns
after subtracting each rule's mean: `log_mean_marginal`, `log_lift`, and
`log_lift × k_rel`. This is the structural question for a model with **fixed
rule intercepts**, not a fitted mixed model. It is not the proposal's pooled
decorrelation gate or a proposal amendment. Zero-variation columns are marked
`no_within_rule_variation`; a coefficient participating in a numerical null
space is `collinear`. The remaining `identified` label concerns only this
specified predictor matrix, not statistical precision, causality, or scientific
validity. Columns are normalized and singular values below `1e-10` times the
largest (with an absolute `1e-12` floor) are treated as zero. Values within
`1e-12` RMS of a rule constant are treated as zero variation.

```sh
uv run python -m exposure.design_identifiability \
  --report outputs/design-diagnostics-57.json \
  --output outputs/design-identifiability-57.json
uv run python -m exposure.design_identifiability \
  --report outputs/design-diagnostics-57-holdout-76001.json \
  --output outputs/design-identifiability-57-holdout.json
```

The exploratory 57-rule pilot has 1,140 rows, centered rank 2 of 3, with
within-rule log-mean-marginal RMS `1.11e-16` and `no_within_rule_variation`.
The other two columns have numerical rank, but do not necessarily provide
adequate statistical precision: their within-rule log-lift variation accounts
for `0.003187/0.245678`, approximately 1.3% of total row-weighted lift
variance, and median per-rule lift range is `0.09553`. Rows share designs and
rules; this is descriptive, not an independent-sample estimate.

The second 100-scene-per-condition measurement matrix gives numerical rank 3
and log-mean-marginal within-rule RMS `0.00308`. That difference is consistent
with deviations on the holdout measurement matrix rather than evidence that
true singleton marginals vary sufficiently: the design was constructed using
the original estimated equality constraints, and Monte Carlo uncertainty is
not propagated. See [design holdout](design-holdout.md). The proposal gate
remains unpassed and this tool deliberately reports `not_assessed`.
