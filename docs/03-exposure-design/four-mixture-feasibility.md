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

## Simultaneous holdout uncertainty

The point violations above are sample estimates, not proof that their
population margins are negative. We now bound their uncertainty with
`exposure.design_factorial_uncertainty`. For a frozen weight vector $w$ and
condition $j$ with $N_j$ independent accepted scenes, a mixture probability is
$\hat p=\sum_j w_j\hat p_j$. Weighted Hoeffding gives
$P(|\hat p-p|>t)\le2\exp[-2t^2/(\sum_j w_j^2/N_j)]$. Applying a union bound
to all four mixtures, six rules, and three events per rule (two singleton
predicates and the conjunction) gives 72 simultaneous mixture-probability
intervals at family-wise 95% coverage, conditional on the frozen design and
the independent-sampling model. The method needs no assumption that the
predicate outcomes within one scene are independent. Interval arithmetic
propagates these bounds through the logarithms and the 72 factor/cross-factor
inequalities; when a probability interval touches zero, the corresponding
log endpoint is unbounded. This intentionally conservative calculation can
leave a constraint unresolved even when its empirical margin is negative.

On the frozen 82675 design and the independently seeded 82682 renderer draw,
**20 of 72 constraints have lower bounds at or above zero, none have upper
bounds below zero, and 52 remain unresolved** at the stated family-wise 95%
level. All 14 negative point margins are among the unresolved constraints.
Thus the strict *sample-level* constraints failed as previously reported;
the intervals neither establish 14 population violations nor certify the
design as feasible. The probability radii are 0.0162–0.0186 across mixtures;
these propagate to wider, asymmetric log-feature intervals. No conclusion
about the broader rule population, model performance, checker or Stage 0
follows. The same pseudorandom generator and renderer were used, not an
external dataset; the sampling model idealizes accepted scenes as independent.

The CLI verifies the original fitting-report hash, matching rule definitions
and order, distinct fitting/holdout seeds and counts consistent with reported
probabilities. A different seed cannot itself prove that the design was
actually frozen before the holdout; consult the protocol above. The result's
SHA-256 is `1d3577673754f621440bf64af12c2219ab616cd75effb26c6c76ad0cec4e7474`.
Reproduce with a new output path:

```sh
OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m exposure.design_factorial_uncertainty \
  --design outputs/factorial-global-82675.json \
  --reference-report outputs/exposure-57-fresh-82511.json \
  --report outputs/exposure-57-fresh-82682.json \
  --output outputs/factorial-uncertainty-82682-new.json
```

## Larger frozen holdout, seed 82916

Before drawing this sample we recorded the frozen 82675 design hash, rule-file
hash, seed 82916, 10,000 accepted scenes per condition and the no-refitting
analysis in `outputs/factorial-82675-holdout-82916-protocol.json` (SHA-256
`70f6b1940a6681e714bb2a7da8ce5ba698c58e2542f7ccf61b2d69c8f54b55f3`).
The 57-condition draw contains 570,000 accepted scenes from 5,784,375
proposals. It is a fresh draw from the same renderer following several earlier
exploratory checks, not external preregistration or an external dataset.

Without refitting, three of the 72 sample-level factorial/cross-factor
constraints were negative; the worst margin was -0.002612. The within-rule
information matrix eigenvalues were 0.02567 and 0.49446, condition number
19.26. The simultaneous weighted-Hoeffding audit certified 33 constraints
met, none violated, and left 39 unresolved at the stated family-wise 95%
level; all three negative point estimates are unresolved, and probability
radii were 0.00512–0.00589. Relative to the earlier
1,000-per-condition draw this is more precise, but the draws differ, and
neither the point-violation counts nor these conservative intervals prove a
population-level comparison. The gate remains open.

```sh
uv run --no-sync python -m exposure.measure_fast --rules outputs/templates-57.json \
  --samples 10000 --seed 82916 --state outputs/fast-measure-82916.state.json \
  --output outputs/exposure-57-fresh-82916.json
OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m exposure.design_factorial_uncertainty \
  --design outputs/factorial-global-82675.json \
  --reference-report outputs/exposure-57-fresh-82511.json \
  --report outputs/exposure-57-fresh-82916.json \
  --output outputs/factorial-uncertainty-82916.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/evaluate-factorial-design.py \
  --design outputs/factorial-global-82675.json --reports outputs/exposure-57-fresh-82916.json \
  --output outputs/factorial-evaluation-82916.json
```

The ignored measurement, uncertainty and evaluation reports have SHA-256
`6ef2e9e6ce8dc1495b78ad89262172d4920f828b0898b826d0ba20a095a6d182`,
`207d9a047d7816edbeea173cdf1c100887b45b4b77a8a7e4e27047d0efca7e55`, and
`e81bfd65cbb0869dbc42ee4f8dead2ac4a8c28f4e988cdd1039a4213dde99342`.
Use fresh output paths when reproducing the commands.

## Larger frozen holdout, seed 82917

After the 82916 result, we recorded the same frozen design and rule hashes,
seed 82917, 100,000 accepted scenes per condition and the same no-refitting
analysis in `outputs/factorial-82675-holdout-82917-protocol.json` (SHA-256
`bbf0d89199ab71efcd676e2963bd9da92fe8b4c8eed08dc7f6e9ea6bfd0c987f`).
The new same-renderer draw contains 5.7 million accepted scenes from
57,819,265 proposals. This is a sequential exploratory sensitivity check,
not an external preregistration, and the 82675 design was not refitted.

The 82917 point estimates gave four negative margins out of 72, with worst
margin -0.0012845. The unregularized information eigenvalues were 0.02558
and 0.48916 (condition number 19.13). The same conservative simultaneous
95% analysis certified 40 constraints met, none violated, and 32 unresolved;
all four negative estimates were unresolved. Probability radii were
0.00162–0.00186. Thus the chosen sample-level constraints still fail, and
the uncertainty bounds neither certify all population constraints nor
establish any population violation. The proposal's benchmark, checker,
population and identifiability concerns also remain open; this is not a
Stage 0 pass. Comparisons with earlier draws are descriptive, not an
independently powered experiment about sample-size effects.

Reproduce the previous section's three commands with seed `82917`, samples
`100000`, and new 82917 output/state paths. The ignored measurement,
uncertainty and evaluation reports have SHA-256
`92042db82bbaa4d2ecf67d745e778505f352f9cf41464c8908285376bf77bddc`,
`d55d4d24683e1e57ac4543585e468ff346caf84a083fd898b380aff536a3bd69`, and
`2cec9b25fde327a1c8c78c95fa60955506da31b8ed73b465785e4feb7003ca7b`.
