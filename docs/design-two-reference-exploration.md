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
