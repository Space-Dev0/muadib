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

## Bounded and multi-matrix follow-up

A subsequent exploratory fit imposed an upper bound of 0.10 on each condition weight. Fitting only seed 82511 met the numerical constraints there and reduced the effective-condition concentration: 11.05–12.75 effective conditions per mixture. When evaluated without refitting on the previously inspected 1,000-scene seeds 82512 and 82513, it violated 18 and 15 constraints, respectively; the worst signed margins were -0.01493 and -0.01567. These are sensitivity results rather than untouched holdout evidence.

The optimizer now also accepts multiple fitting matrices, constrains each separately, and maximizes their mean ridge-stabilized log determinant. A three-start fit with cap 0.10 to both seed 82511 and seed 82512 satisfied all specified constraints numerically on those two matrices. Evaluation without refitting on the previously inspected seed 82513 still violated 15 constraints, with worst signed margin -0.01601. The maximum condition weight was 0.10, effective-condition counts were 10.92–12.69, and the unregularized within-rule information matrix's condition number on that draw was 16.08. Its determinant is scale-dependent and cannot establish statistical power. Both the contrast thresholds and the six-rule subset remain exploratory. No uncertainty intervals, novel holdout, or optimized model-training experiment are implied by these results.

Reproduce the follow-up with fresh output paths:

```sh
OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m exposure.design_factorial \
  --report outputs/exposure-57-fresh-82511.json \
  --fit-reports outputs/exposure-57-fresh-82512.json \
  --rules attr2_rel0_attributes_1 attr2_rel0_attributes_2 attr2_rel0_attributes_3 \
          attr2_rel0_attributes_4 attr2_rel0_attributes_5 attr2_rel0_attributes_6 \
  --starts 3 --maxiter 200 --max-weight 0.1 --delta-m 0.02 \
  --delta-l 0.02 --cross-tolerance 0.05 --output outputs/factorial-two-fit-new.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/evaluate-factorial-design.py \
  --design outputs/factorial-two-fit-new.json \
  --reports outputs/exposure-57-fresh-82511.json outputs/exposure-57-fresh-82512.json \
            outputs/exposure-57-fresh-82513.json \
  --output outputs/factorial-two-fit-check-new.json
```

The original two-matrix fit JSON was SHA-256 `e14e1d337c13eab3766c419b64e6c6b5b59f57eb0d88a6ecaa7e1393102ab755`. The validation CLI checks rule/condition compatibility, but the 82513 draw has been inspected before, so the next research step is a frozen candidate evaluated on genuinely new draws, with uncertainty and sampling-budget constraints.

## Frozen check on a fresh renderer draw

Before generating seed 82681, we recorded the SHA-256 of the selected two-fitting-matrix design (`e14e1d337c13eab3766c419b64e6c6b5b59f57eb0d88a6ecaa7e1393102ab755`), the seed and 1,000 accepted scenes per condition in the ignored `outputs/factorial-82673-fresh-protocol.json`. Without refitting, the new exposure report (SHA-256 `6b827f08651440c722ab8353be719015aa342e17f2f2e2bfcd0790b7962619e5`) yielded eight factorial/cross-factor constraint violations, worst margin -0.01155, and unregularized information condition number 13.93. The new draw uses the same renderer, is not an external benchmark, and does not quantify Monte Carlo uncertainty. It fails the chosen strict sample-level constraints; no claim about population infeasibility or the full research gate follows. The completed non-refitting check is `outputs/factorial-global-82673-fresh-82681.json` (ignored). A separate reference under simplified iid-object assumptions is documented in `docs/01-instruments/object-count-baseline.md`.

A further candidate was fitted on all three previously inspected matrices (82511, 82512 and 82513), with the same 0.10 weight cap and factorial thresholds. Its SHA-256 `908a8ac581a318cfecfeba2d061d652a1100c34bda6d108cf4b1f77f59ef4dda` and an evaluation protocol for seed 82682 were recorded before generating that new draw. All three local optimizer starts reported feasible fits on the three fitting matrices. On the fresh 82682 renderer draw (1,000 accepted scenes per condition, report SHA-256 `2702240c61951ebc06bc8af6041ed15fc37f71b7200d0e40105d68b62854bb40`), the frozen design had **14 constraint violations**, worst signed margin -0.01081 and information condition number 19.69. It used at most 0.10 weight per condition and had 11.46–15.20 effective conditions per mixture. This is another same-renderer, finite-sample sensitivity test, not evidence that the underlying population is infeasible or that further fitting alone will solve the issue. Protocol, frozen design and results remain in ignored `outputs/` with the `82675`/`82682` identifiers.
