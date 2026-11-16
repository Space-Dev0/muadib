# Exploratory bounded rule allocation

The previous endpoint design experiment sampled each of 57 rules equally and
still had held-out lift VIF 2.942. This diagnostic asks a different question:
can a **nonuniform allocation of evaluation effort across rules** reduce the
pooled correlations while retaining every rule? This changes the rule-weighted
analysis population. It is an exploratory option for research review, not a
Stage 0 gate pass or a validated experiment.

Fit the previously explored nonuniform reference on the 100-scene-per-condition
pilot matrix, and generate its reference plus 114 full-rule LP endpoints with
minimum source weight 0.0001. Assign every candidate design equal mass. Fit one
probability per rule on the pilot matrix, shared across designs, subject to each
rule receiving between 0.2 and 2 times its equal-allocation mass and all rule
probabilities summing to one. Local SLSQP minimizes squared VIF excess above
1.75 and squared absolute-correlation excess above 0.25 for the existing four
features. These are exploratory optimization targets, not proposal thresholds
or evidence of global optimality. Freeze the resulting rule allocation and
endpoint designs and recompute the same statistics with the independently
measured 500-scene-per-condition matrix. Each entry is a weighted pooled
regression including an intercept; rule-by-design observations are dependent.

| Allocation | Matrix | r(log marginal, log lift) | VIF marginal | VIF lift | VIF attributes | VIF relations |
|---|---|---:|---:|---:|---:|---:|
| Equal rules | Pilot | 0.012 | 1.905 | 2.766 | 1.620 | 3.027 |
| Equal rules | Holdout | −0.005 | 1.922 | 2.942 | 1.683 | 3.190 |
| Fitted rules | Pilot | 0.118 | 1.494 | 1.865 | 1.387 | 1.823 |
| Frozen fitted rules | Holdout | 0.100 | 1.485 | 1.911 | 1.423 | 1.848 |

An integer illustration uses largest-remainder rounding of the fitted rule
weights to 285 observations for each candidate design. That gives 1–10
observations per rule per design and a total of 32,775 observations across the
115 candidate design entries; the held-out VIFs are 1.485, 1.916, 1.422, and
1.841, and correlation 0.111. This example demonstrates a realizable
allocation for the proxy matrix, **not** adequate statistical power, acceptable
cost, or a requirement to train these models. An actual allocation should be
pre-specified before data collection and account for uncertainty and run cost.

The candidate is highly concentrated: 31/57 rule probabilities hit the lower
bound, 23 hit the upper bound, and effective rule count `1/sum(q²)` is about
32.1. Of the 115 endpoint entries, only 104 have distinct source mixtures at
nine decimal places; treating duplicated entries equally also weights some
unique mixtures more heavily. A source mixture gives one condition up to 0.544
weight, and the fitted reference's original 0.04 cap is not imposed on
endpoints. Positivity of all weights does not make these designs practically
trainable or guarantee meaningful exposure contrasts. The two report matrices
are Monte Carlo estimates; the holdout is a second measurement of the same
rules, not external benchmark validation. No uncertainty is propagated.

The VIF calculation omits proposal breadth and treats rule-by-design rows as
independent for its descriptive algebra. It does not analyze a final regression
or accommodate correlations among outputs from a shared trained model. The
log marginal remains fixed within each rule on the fitting matrix, so its
coefficient is still unidentifiable alongside rule fixed effects. Reweighting
rules does not solve that structural identification problem. This is a
post-hoc, selectively fitted allocation; the proposed equal-rule estimand must
not be silently replaced. A future protocol must justify the target population,
minimum per-rule representation, training cost, all covariates, independent
replications, checker reliability, and independent exposure measurements.
Optimal-design research illustrates how allocating experimental effort changes
an information matrix under a chosen model; it does not establish that our
allocation meets the assumptions or solves these problems. See [optimal design
under correlated observations](https://arxiv.org/abs/1303.2863).

Reproduce with the existing local measurement reports:

```sh
OPENBLAS_NUM_THREADS=1 uv run python -m exposure.design_rule_allocation \
  --training-report outputs/exposure-57-pilot.json \
  --holdout-report outputs/exposure-57-holdout-81205.json \
  --output outputs/design-rule-allocation-new.json
uv run python -m unittest tests.test_design_rule_allocation -q
```

The command refuses to overwrite its output and includes input hashes, frozen
source mixtures, rule weights, and the integer example. The exploratory local
report is `outputs/design-rule-allocation-explore-v3.json` (ignored by Git).
