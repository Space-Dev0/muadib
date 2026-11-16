# Exploratory design correlation by rule and complexity

`exposure.design_strata` reads an existing `design_diagnostics` report. It
compares the pooled Pearson correlation of log mean marginal exposure and log
lift with the correlation after separately subtracting each rule's mean. It
also reports correlation between rule means, replicated once per retained row
to preserve the pooled row weighting, and correlation within each
attribute-count/relation-count stratum.

```sh
uv run python -m exposure.design_strata --report outputs/design-diagnostics-57.json --output outputs/design-strata-57.json
uv run python -m exposure.design_diagnostics --report outputs/exposure-57-holdout-76001.json --designs outputs/design-samples-57.json --output outputs/design-diagnostics-57-holdout-76001.json
uv run python -m exposure.design_strata --report outputs/design-diagnostics-57-holdout-76001.json --output outputs/design-strata-holdout-76001.json
```

The output of `design_strata` must be a new file. It requires finite rows,
unique rule-by-design pairs and consistent complexity for each rule. It
returns `null` for a correlation with too few rows or essentially constant
features. It does not recreate excluded rows; review exclusion counts in the
input report. A stratum can contain many rows from only a few rules, not
independent observations.

For the 57-rule, 20-design exploratory pilot, pooled correlation was -0.4461
on the original matrix. The rule-mean correlation with the same row weighting
was -0.4491. Within-rule centered correlation is **undefined** there: the
estimated singleton-preserving equality makes the mean marginal essentially
constant for each rule, leaving only numerical precision. Treating that
rounding noise as variation can produce a spurious correlation.

On the independently seeded 100-scene-per-condition matrix, keeping the same
designs, pooled correlation was -0.4640, the row-weighted rule-mean correlation
was -0.4671 and the within-rule correlation was -0.00092. The within-rule
variation there reflects deviations on the holdout matrix; it does not imply
that those designs preserve true marginals exactly. For example, the maximum
singleton deviation was 0.01767; see [holdout](design-holdout.md). The
single-attribute/no-relation stratum has constant log lift and an undefined
correlation on both matrices.

These descriptive values show how correlations change with aggregation; they
do not establish why correlations arise, quantify uncertainty, account for
dependence, decide whether the proposal's breadth covariate is appropriate, or
pass/fail the Stage 0 gate. Designs were selected using the first estimated
matrix. Different retained-row patterns would change the weighting, so compare
exclusions before comparing matrices. Reports under `outputs/` are ignored.
