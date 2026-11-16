# A diagnostic lower bound for VIF

`exposure.design_vif_floor` checks a specific structural obstacle in the
existing exploratory pooled design. It requires complete rule-by-design rows,
equal rule weighting, constant mean marginal exposure within each rule, and
variation in both marginal exposure and relation count across rules. It is
not the proposal's gate evaluation or a random-effects model.

For either of these predictors, VIF is `1/(1-R²)`, where `R²` comes from the
auxiliary regression on all the other predictors and an intercept. Including
the other one of this pair alone gives `R² = r²`; adding more regressors cannot
reduce the fitted `R²`. Therefore the VIF for each is at least `1/(1-r²)`.
If both are retained as predictors, no choice of extra predictors can bring
either VIF below that lower bound on the same rows.

For the seed-43 57-rule estimated matrix, the correlation between log mean
marginal and relation count is 0.84409. Its lower bound is **3.4781**, already
above 2, even before adding lift or other covariates. Since marginal exposure
and relation count are constant within each rule on this matrix, repeating the
same complete rule set for more singleton-preserving designs leaves this
particular correlation unchanged. The same value appears for both the original
20 designs and the 115 LP endpoint candidates. This does not prove that every
possible benchmark, weighting, model specification, or population exposure
has the same limitation. Incomplete rows, unequal weights, approximate
constraints and Monte Carlo error also change the calculation. Do not remove or
redefine a proposal criterion silently; benchmark and analysis choices must be
documented before any gate claim.

```sh
uv run python -m exposure.design_vif_floor \
  --report outputs/design-diagnostics-57.json --output outputs/design-vif-floor-57.json
```
