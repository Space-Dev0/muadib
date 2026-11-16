# Exploratory nonuniform-reference design diagnostics

This analysis asks whether the candidate reference mixture in
[reference-mixture sensitivity](design-reference-exploration.md) helps the
original pooled decorrelation diagnostic after actual singleton-preserving
mixtures are sampled. This is not a gate pass. The fitted reference changes
the exposure population; the fit and support bounds were chosen exploratory.
The fitted weights are constrained to [0.001, 0.04] and many sit at these
bounds. Each hit-and-run chain starts at that fitted reference, enforces only
nonnegative weights (not the 0.04 cap), and uses 1,000 burn-in steps and ten
steps between 100 retained designs. There are no mixing diagnostics or proof
of independent designs. The prior claimed pairwise VIF floor was not a complete
VIF analysis; this analysis includes the current four diagnostic features.

For each of five chain seeds 81224–81228, we evaluate the same designs on the
100-scene/condition seed-43 fitting matrix and a separately generated
500-scene/condition seed-81205 holdout matrix. Diagnostics use complete
rule-by-design rows and include log mean marginal, log lift, attribute count,
and relation count; breadth remains omitted. As a result even a favorable value
would not verify the full proposed gate. The holdout measures Monte Carlo
sensitivity, not training-run uncertainty or a new benchmark, and the same
rule-by-design rows are dependent. Variation in singleton marginals on the
holdout is an estimated out-of-sample deviation from each report's own reference
mixture, not a guarantee about true exposure constraints.

| Chain seed | Holdout r(log marginal, log lift) | VIF marginal | VIF lift | VIF attributes | VIF relations | Max marginal deviation |
|---:|---:|---:|---:|---:|---:|---:|
| 81224 | -0.0300 | 2.464 | 6.233 | 2.804 | 6.032 | 0.002691 |
| 81225 | -0.0298 | 2.441 | 6.111 | 2.762 | 5.925 | 0.002061 |
| 81226 | -0.0356 | 2.449 | 6.218 | 2.782 | 6.049 | 0.001696 |
| 81227 | -0.0271 | 2.424 | 5.997 | 2.733 | 5.809 | 0.004166 |
| 81228 | -0.0266 | 2.462 | 6.219 | 2.810 | 5.995 | 0.002298 |

For the five fitting-matrix calculations, correlations range from -0.0078 to
0.0008 and lift VIF from 5.613 to 5.870; the largest estimated constraint
residuals round to 0.000000. For seed 81224, no rows were excluded from the diagnostics.
The original proposal's absolute correlation target of 0.3 is satisfied in
these descriptive samples, but the observed lift VIF exceeds 2 on both matrices
in every chain. Other reported VIFs also exceed 2. Thus this intervention alone
does not achieve the design criterion. Within-rule mean marginal remains fixed
on the fitting matrix, so its coefficient is collinear with rule intercepts;
this proposed mixture does not identify that effect.

Reproduce from the repository root with the measurement reports above and the
project dependencies installed:

```sh
uv run python - <<'PY'
import json
import numpy as np
from exposure.design_reference_explore import matrix, optimize_reference
from exposure.design_sample import sample_designs
from exposure.design_diagnostics import diagnose
from exposure.mixture import summarize

pilot = json.load(open('outputs/exposure-57-pilot.json'))
holdout = json.load(open('outputs/exposure-57-holdout-81205.json'))
names, values, counts = matrix(pilot)
weights = dict(zip(names, map(float, optimize_reference(values, counts))))

def marginals(report, mixture):
    result = summarize(report, mixture)['results']
    return np.array([v for item in result.values() for v in item['predicate_marginals']])

for seed in (81224, 81225, 81226, 81227, 81228):
    designs = sample_designs(pilot, weights, samples=100, seed=seed,
                             min_weight=0.0, burn_in=1000, thin=10)
    for label, report in (('pilot', pilot), ('holdout', holdout)):
        diagnostics = diagnose(report, designs)
        reference = marginals(report, weights)
        deviation = max(float(np.max(np.abs(marginals(report, design['weights']) - reference)))
                        for design in designs['designs'])
        features = ('log_mean_marginal', 'log_lift', 'k_attr', 'k_rel')
        print(seed, label,
              diagnostics['correlations']['log_mean_marginal']['log_lift'],
              [diagnostics['vif'][key]['value'] for key in features],
              'excluded', diagnostics['excluded_count'], 'max deviation', deviation)
PY
```
