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
