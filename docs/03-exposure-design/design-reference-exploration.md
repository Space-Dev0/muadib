# Exploratory reference-mixture sensitivity

This is a diagnostic intervention, not a Stage 0 pass. In the existing balanced,
complete rule-by-design table, each rule's mean singleton marginal is fixed by
the reference mixture. The lower bound in `design-vif-floor.md` therefore does
not improve by adding more singleton-preserving designs for the same reference.
Here we ask whether **changing the reference mixture itself**, within explicit
weight limits, changes that lower bound. This changes the population relative
to which exposure is defined and requires a documented research decision before
it can replace the original design.

For each rule and source condition, calculate the mean of its singleton-event
probabilities. Multiply that matrix by a mixture vector, then correlate the log
of the resulting mean marginal over the equally weighted rules with their
relation counts. Fit the mixture on the seed-43 pilot matrix by minimizing the
squared correlation, subject to the weights summing to one and lying in
[0.001, 0.04]. Freeze those weights and evaluate them on the independently
measured 500-scene-per-condition seed-81205 matrix. SLSQP is a local optimizer;
this is not a proof of the global optimum or a prespecified confirmatory study.
The bounds were chosen for this exploration and are not a proposal requirement.

| Measurement | Correlation | Pairwise VIF lower bound |
|---|---:|---:|
| Pilot, uniform reference | 0.8441 | 3.478 |
| Pilot, fitted reference | 0.5709 | 1.484 |
| Independent matrix, uniform reference | 0.8434 | 3.465 |
| Independent matrix, frozen fitted reference | 0.5856 | 1.522 |

The pairwise bound follows because the auxiliary regression for either predictor
includes the other and an intercept: additional regressors cannot reduce R²,
so VIF ≥ 1/(1-r²). The lower bound falling below 2 removes this *particular*
proof that the VIF must exceed 2. It does **not** show actual VIF < 2, or that
the separate proposal constraint on correlation of log mean marginal with log
lift is satisfied. Both must be evaluated using actual complete design rows and
all proposal predictors, with the proposal thresholds unchanged.

The fitted mixture is highly concentrated despite the cap: 32 of 57 weights
are at the minimum and 24 at the maximum (numerical tolerance 1e-6). Such a
reference has an altered scientific interpretation and requires review; the
holdout only measures sensitivity to a second Monte Carlo matrix, not validation
on a fresh benchmark. For singleton-preserving designs, mean marginal exposure
still does not vary within a rule, so a within-rule marginal coefficient remains
unidentified with rule intercepts. This approach does not establish learnability,
checker validity, or human audit success.

Reproduce (the output is exploratory and may be regenerated):

```sh
uv run python -m exposure.design_reference_explore \
  --training-report outputs/exposure-57-pilot.json \
  --holdout-report outputs/exposure-57-holdout-81205.json \
  --output outputs/design-reference-exploration.json
uv run python -m unittest tests.test_design_reference_explore -q
```
