# Exploratory global four-mixture feasibility

`exposure.design_factorial` implements a first CPU-only check of the review's suggestion. It is an exploratory design investigation, not an approved estimand, a reproducible model comparison, a training authorization, or a Stage 0 pass.

The code selects two-predicate target rules, loads estimated singleton and full-rule probabilities by condition, and constructs four shared simplex mixtures LL/LH/HL/HH. For each rule it uses the **mean of log singleton probabilities** as marginal feature and `log joint - sum(log singleton)` as lift. This differs from the `log(mean singleton)` feature in some earlier design diagnostics. Rules have equal weight in the information matrix. SLSQP maximizes a ridge-stabilized log determinant of the within-rule-centered 2×2 feature matrix. It imposes a minimum weight and, for every selected rule, marginal-factor and lift-factor contrasts of at least 0.02 on both matched pairs. It bounds each cross-factor feature difference in absolute value by 0.05. These numbers are exploratory choices, not empirically justified thresholds. The same condition weights apply across all selected rules; no rule-specific retraining is implied by this construction.

The initial example uses the existing 100,000-scene-per-condition seed-82511 matrix and six `attr2_rel0_attributes_1` through `attr2_rel0_attributes_6` rules. This is an easily reproduced subset of the existing provisional benchmark, **not** a complete or representative rule population and not a binding-validated confirmatory target. With floor 0.001, three random starts and 200 maximum iterations, all three runs reported success at numerical tolerance. The best fitting matrix had information eigenvalues 0.0468 and 0.5264 (condition number 11.25). Its largest single-condition weight was 0.4224; effective condition counts `1/sum(w²)` across the four mixtures were 3.42, 5.56, 4.02 and 7.06 out of 57. This is a highly concentrated training-distribution change.

The selected weights were then evaluated without refitting on existing 1,000-scene-per-condition seed-82512 and seed-82513 matrices. Both had already been used in prior exploratory analysis, so these are sensitivity checks **not untouched confirmatory holdouts**. Seed 82512 violated 13 of the factor/cross-factor constraints (worst signed margin -0.0446); seed 82513 also violated 13 (worst -0.0721). Their information-matrix condition numbers were 13.49 and 11.95. Thus the strict empirical factorial constraints are not robust to these measurement draws. Neither a nonsingular feature matrix nor local solver success establishes causal identification, sufficient statistical power, or generalization. A fresh, frozen candidate must be validated with new seeds and quantified Monte Carlo uncertainty before any training decision.

The solver checks input shape and probabilities, validates nonnegative weights and unit sums, checks every declared contrast after solving, and reports raw eigenvalues and per-start optimizer status. Its determinant includes a small ridge to permit optimization near singular starts; feasibility checks use the **unregularized** eigenvalues. It requires strictly positive mixed probabilities for all logarithms; zeros are not silently clipped. SLSQP and three starts cannot prove global optimality or infeasibility. The procedure does not yet include all singleton marginal-preservation constraints, confidence intervals, sampling-budget constraints, uniform stability limits or within-rule recovery simulation. In particular it should not be presented as completing G2.

Reproduce with existing ignored measurement reports:

```sh
OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m exposure.design_factorial \
  --report outputs/exposure-57-fresh-82511.json \
  --rules attr2_rel0_attributes_1 attr2_rel0_attributes_2 attr2_rel0_attributes_3 \
          attr2_rel0_attributes_4 attr2_rel0_attributes_5 attr2_rel0_attributes_6 \
  --starts 3 --maxiter 200 --delta-m 0.02 --delta-l 0.02 --cross-tolerance 0.05 \
  --output outputs/factorial-global-new.json
```

For a previously fitted JSON, run the non-refitting evaluator with new output names:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/evaluate-factorial-design.py \
  --design outputs/factorial-global-new.json \
  --reports outputs/exposure-57-fresh-82511.json outputs/exposure-57-fresh-82512.json \
            outputs/exposure-57-fresh-82513.json \
  --output outputs/factorial-global-sensitivity-new.json
```

The JSON records the input SHA-256, rule and condition orders, settings, all per-start results, weights, eigenvalues and numerical violations. Never overwrite or reinterpret previous exploratory outputs. Analytic gradients are checked against finite differences in `tests/test_design_factorial.py`.
