# Two-reference exploratory exposure design

The existing singleton-preserving design fixes mean marginal exposure within
rule, which makes its marginal coefficient collinear with rule intercepts. This
experiment combines endpoint designs built around two different reference
mixtures: uniform and the exploratory fitted nonuniform reference. Each design
still preserves its own reference's estimated singleton marginals, but the two
references have different targets. This changes the original fixed-marginal
setup and scientific interpretation; it is not a proposal gate pass.

For 115 endpoint entries per reference, centered within-rule log marginal and
log lift have rank 2 on both matrices, with within-rule correlations 0.226
(pilot) and 0.237 (holdout). This removes the specific exact collinearity in
that two-variable centered matrix. It does not prove identification in the
full planned model or after adjustment for reference group and other predictors.
The held-out pooled VIFs remain 2.162 marginal, 2.496 lift, 1.556 attribute
count and 3.280 relation count. Thus the proposal's VIF < 2 condition fails
for this exploratory allocation. Breadth, dependence and measurement
uncertainty remain omitted; reference mixtures and extreme endpoints alter the
population and may be costly to train. The independent matrix is a separate
Monte Carlo measurement, not a new external benchmark.

### Rule-by-reference adjustment

With an intercept for each rule and reference combination, the same endpoint
designs do **not** identify the marginal-exposure coefficient on the fitting
matrix. Within a reference the design constraints preserve the singleton
marginals, so the mean marginal for a given rule is constant over designs;
its log is therefore absorbed by the rule-by-reference intercept. Numerically,
across 114 rule-reference groups and 13,110 rows the within-group marginal
root-mean-square deviation is `8.53e-15`, the largest group range is
`2.36e-13`, and the centered marginal-and-lift matrix has rank 1 at tolerance
`1e-10`. The within-group lift RMS is `0.399`.

On the independent Monte Carlo matrix the corresponding marginal RMS is
`0.0141`, maximum range `0.220`, and numerical rank 2. The designs only
preserve singleton constraints on the fitting estimates; deviations on an
independent estimate can reflect Monte Carlo measurement and constraint
error. That numerical rank therefore does not by itself demonstrate real,
reliably measured within-cell marginal variation. This diagnostic addresses
the specified rule-by-reference intercept model; it does not prove that every
other adjustment, such as a shared reference intercept, is unidentifiable.
For example, after removing rule intercepts and one shared reference intercept,
the two residualized exposure columns have rank 2 on both matrices; their
correlations are 0.227 and 0.238, with corresponding two-column residual VIFs
1.054 and 1.060. These are equal-row exploratory diagnostics, not VIFs for the
full planned model or a test of the reference-intercept assumption. Breadth,
measurement uncertainty and the choice of estimand remain unresolved.

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/explore-reference-cell-identifiability.py
```

Reproduce using the existing ignored matrix reports:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run python scripts/explore-two-reference.py
```

A further exploratory fit reweights rules across both reference groups using
`design_rule_allocation.fit`, with every rule between 0.2 and 2.0 times equal
weight. On the fitting matrix it produces pooled correlation 0.173 and VIFs
1.621, 1.848, 1.471 and 1.819; the frozen weights and designs on the holdout
produce correlation 0.161 and VIFs 1.629, 1.908, 1.507 and 1.863, respectively.
All are below the proposal's numerical thresholds for these four features.
The effective rule count is approximately 32.4 of 57; this reweights the rule
population and remains post-hoc. Breadth and dependence are omitted, the
reference-group adjustment and full-model identification remain untested,
the independent matrix is not an external benchmark, and the endpoint mixtures
may be impractical. These numbers are a promising exploratory candidate, not a
Stage 0 pass or authorization to change the research estimand.

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run python scripts/explore-two-reference-allocation.py
```
