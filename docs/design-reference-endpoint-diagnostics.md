# Exploratory endpoint designs for the nonuniform reference

The hit-and-run designs in [the nonuniform reference diagnostics](design-reference-design-diagnostics.md)
produce pooled lift VIF near six. One contributor is that log lift correlates
with both complexity features: for chain seed 81224, `r(log lift, k_attr)` is
−0.590 and `r(log lift, k_rel)` is −0.621, even though
`r(log mean marginal, log lift)` is −0.00455. These are exploratory pooled
correlations, not causality or independent observations; the VIF also depends
on the combined predictors.

A candidate intervention is to maximize or minimize each rule's full-rule
exposure with an LP, while preserving all estimated singleton marginals of the
fitted nonuniform reference. We used `exposure.design_extremes.generate`, which
includes the reference plus both endpoints for each of 57 rules (115 designs).
Each design has a minimum weight bound; the fitted reference's 0.04 maximum is
not enforced for endpoint designs. This exploits extreme design points, so the
results do not show that such distributions are scientifically desirable, that
training is feasible, or that a sampler mixes well. The reference fitting and
these endpoint choices are exploratory and alter the exposure population.

| Minimum weight | Measurement matrix | r(log marginal, log lift) | Lift VIF | Relation VIF |
|---:|---|---:|---:|---:|
| 0.0001 | Pilot | 0.0122 | 2.766 | 3.027 |
| 0.0001 | Holdout | −0.0047 | 2.942 | 3.190 |
| 0.0005 | Pilot | 0.0073 | 2.922 | 3.170 |
| 0.0005 | Holdout | −0.0101 | 3.065 | 3.308 |
| 0.001 | Pilot | 0.0009 | 3.048 | 3.292 |
| 0.001 | Holdout | −0.0174 | 3.180 | 3.425 |
| 0.00000001 | Pilot | 0.0136 | 2.706 | 2.973 |
| 0.00000001 | Holdout | −0.0034 | 2.910 | 3.160 |

There were no excluded rows in these runs. At floor 0.0001, pilot/holdout
marginal VIF was 1.905/1.922 and attribute VIF was 1.620/1.683; all four
features (marginal, lift, attribute count, relation count) were included. The
smaller floor can further concentrate training weights and does not solve the
VIF gate. Breadth is still omitted, correlations and VIFs ignore row dependence,
and the independent measurement matrix only probes Monte Carlo sensitivity.
Thus endpoint designs improve the descriptive VIF but **do not pass the
proposal's VIF < 2 condition**, much less establish the full gate. Equality of
marginals applies exactly to the estimated pilot matrix, not automatically to
true exposures or the independent matrix. The within-rule marginal coefficient
still cannot be identified alongside fixed rule intercepts.

Reproduce (the script can take a minute; `OPENBLAS_NUM_THREADS=1` makes timing
more predictable):

```sh
OPENBLAS_NUM_THREADS=1 uv run python - <<'PY'
import json
from exposure.design_reference_explore import matrix, optimize_reference
from exposure.design_extremes import generate
from exposure.design_diagnostics import diagnose

pilot = json.load(open('outputs/exposure-57-pilot.json'))
holdout = json.load(open('outputs/exposure-57-holdout-81205.json'))
names, values, counts = matrix(pilot)
reference = dict(zip(names, map(float, optimize_reference(values, counts))))
for floor in (0.0001, 0.0005, 0.001, 0.00000001):
    designs = generate(pilot, reference, min_weight=floor)
    for label, report in (('pilot', pilot), ('holdout', holdout)):
        result = diagnose(report, designs)
        print(floor, label,
              'r', result['correlations']['log_mean_marginal']['log_lift'],
              'vif', result['vif'], 'excluded', result['excluded_count'])
PY
```
