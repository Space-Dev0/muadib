# Exploratory measurement holdout for fixed exposure designs

`exposure.design_holdout` evaluates previously selected designs against a second
subset-enabled measurement matrix. It reports each design's singleton-exposure
deviation from the same reference mixture separately on both matrices, along
with the existing pooled correlations, VIFs and row-exclusion counts. The
designs are held fixed; the tool does not refit them on the second matrix.

For the provisional 57-rule pilot, generate a new matrix and evaluate the 20
designs already sampled from the seed-43 matrix:

```sh
uv run python -m exposure.measure --rules outputs/templates-57.json --samples 100 --seed 76001 --include-subsets --output outputs/exposure-57-holdout-76001.json
uv run python -m exposure.design_holdout --training outputs/exposure-57-pilot.json --holdout outputs/exposure-57-holdout-76001.json --reference outputs/uniform-57.json --designs outputs/design-samples-57.json --output outputs/design-holdout-76001.json
```

The comparison requires matching rule definitions and order, subset-enabled
reports with distinct recorded integer seeds, valid reference weights and
designs. It refuses to overwrite its output. A different seed is a useful
separation, not proof that sampled scenes are independent or disjoint. Input
generation and candidate design sampling are described in
[exposure measurement](exposure-measurement.md) and
[design sampling](design-sample.md).

In this exploratory run, each condition had 100 accepted scenes in each
matrix. All 1,140 rule-by-design rows were retained in both matrices. Pooled
log-mean-marginal/log-lift correlation was -0.4461 on the original matrix and
-0.4640 on the new matrix; log-lift VIF was 11.61 and 11.07, and relation-count
VIF was 13.72 and 13.70, respectively. The largest absolute singleton deviation
from the uniform reference across the 20 designs and singleton events was
approximately zero on the original matrix and 0.01767 on the new matrix.

These numbers describe one additional Monte Carlo matrix, not an uncertainty
interval or a registered gate analysis. Rows share rules and designs; breadth
is omitted; sampling and design-selection uncertainty remain unresolved. The
observed deviations show why equality on the original estimated matrix does
not guarantee equality on an independently seeded estimate. They do not prove
that the proposal's decorrelation gate has passed or failed for the final
benchmark. Reports under `outputs/` are ignored by Git.
