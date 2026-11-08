# Suspicious Implementation Patterns & Research Risk Heuristics

> **Correction status (2026-11-05):** The original claim that the proposal's
> `Γ = logit(RSR) − logit(Π c_p)` is negative under the multiplicative null is
> retracted. `logit` is injective, so the stated gap is exactly zero when
> `RSR = Π c_p`; the original `−2.20/−4.11` numbers belong to
> `logit(Π c_p) − Σ logit(c_p)`. The items below now use independently verified
> values. See `docs/verification-2026-11-04.md` for proofs and
> `exposure/composition.py` for the corrected implementation.

> [!NOTE]
> These are prioritized research and implementation risks, not claims that the
> project is fraudulent. Status marks whether a code-level mitigation exists.

---

## 1. Methodological & Theoretical Risks

### A. Composition-Gap Scale and the Multiplicative Null — *partly resolved*

- **Observation:** The proposal defines
  `Γ_r = logit(RSR_r) − logit(Π_{p∈P_r} c_p)`.
- **Corrected analysis:** For `RSR = Π c_p` with all probabilities strictly
  inside `(0,1)`, both `Γ^logit` and `Γ^log = log RSR − Σ log c_p` are exactly
  zero. The two are different transforms of the same deviation, not competing
  nulls. The original review's table evaluated `logit(Π c_p) − Σ logit(c_p)`.
- **Residual risk:** (i) unknown or fixed design choice between logit and log
  scales; (ii) boundary values (`0`, `1`) make the logit undefined and can
  silently drop saturated rules; (iii) the product null ignores shared object
  bindings, so it is not a coherent joint distribution when predicates share
  variables.
- **Status:** `exposure/composition.py` reports both scales plus an explicit
  `gamma_status`; `tests/test_composition.py` verifies the null and the
  boundary behavior. The binding-aware maximum-entropy null is **not**
  implemented.
- **Recommendation:** Pre-register one scale as primary, report both, and
  develop the binding-aware null before making composition claims.

### B. Relational Predicate Saturation & Endpoint Confounding — *open, blocking*

- **Observation:** `left_of`/`above` are existential. With three objects on the
  3×3 grid plus jitter, `P(all x equal) = (3/C(9,3))·(1/49) = 1/1372`.
- **Verified values:** analytic singleton exposure `0.999271`; 200,000-scene
  simulation `0.99930` (`left_of`) and `0.99916` (`above`); saved matrix
  `3,249` measurements with min `0.99`, median `1.00`.
- **Residual risk:** A relation singleton cannot separate relational competence
  from arbitrary placement. Relation kind is also confounded with endpoint
  binding: in `K_attr=2, K_rel=1`, `left_of` is subject-only `3/3` and `above`
  is both-bound `3/3`. Only `3/12` color–shape pairs are co-bound to the same
  object.
- **Status:** `exposure/benchmark_audit.py` detects and reports all three
  failures (`saturated_relation_singletons`,
  `relation_kind_binding_confound`, `incomplete_same_object_pair_coverage`)
  and sets `confirmatory_ready: false`. The benchmark itself is **not fixed**.
- **Recommendation:** Bind both relation endpoints to attributed objects,
  counterbalance relation kind within cell, cover all `12` same-object pairs,
  and target relation singletons in `[0.2, 0.8]`.

### C. Sampled Dose Is Narrow and One-Sided; Feasible Set Is Wider — *partly resolved*

- **Observation:** Across 20 sampled singleton-preserving designs (`1,140`
  rule-by-design rows), all lifts are `≤ 0`; median per-rule range `0.0955` log
  units; `98.7%` of lift variance is between-rule.
- **Verified exact bounds:** LP over the singleton-preserving polytope gives
  lift in `[−4.888, +0.337]`, median width `1.172`; `23/57` rules admit positive
  lift. A 5-chain × 200-design run still gives median range `0.148–0.182` and
  `0/57,000` positive lifts. The positive corner has `47/57` weights at the
  floor.
- **Residual risk:** `β_λ` and `β_λR` are weakly identified by the current
  sampler, and the positive arm is effectively unreachable. This is a
  sampler/design-coverage failure, not proof that the instrument cannot express
  positive lift.
- **Status:** `exposure/design_bounds.py` computes exact LP bounds and the
  positive corner; `exposure/design_chains.py` reports split-R-hat and
  autocorrelation ESS. Neither is wired into the default sampling path, and the
  coefficient-targeted design is **not** implemented.
- **Recommendation:** Target `β_λ`/`β_λR` directly (D-optimal or c-optimal),
  report exact bounds alongside every sample, and drop rules whose attainable
  range is below a pre-registered threshold.

### D. Model Learnability & Stage 0 Gate — *open, blocking*

- **Observation:** The 12,000-update online samples were noise. The
  32,000-update run produced two inspected samples: a red shape for the red
  condition and no clear requested relation. The fixed-scene diagnostic had
  MSE `0.8017` on `[0,1]`.
- **Residual risk:** Stage 0 has no model criterion. A run can "pass" the
  checker, design, and MOSAIC gates with no usable generator, making every
  downstream `RSR`, `N50`, and `Γ` uninterpretable.
- **Status:** No learnability pre-gate is implemented and no Stage 1 pilot
  exists.
- **Recommendation:** Add overfit-1/10/100-scene and full-template gates; report
  a learnability failure as the result rather than proceeding.

### E. Decorrelation Gate and the Between-Rule Artifact — *open, blocking*

- **Verified values:** pooled `r(log m̄, log lift) = −0.446` (pilot) and
  `−0.464` (holdout); VIF `11.61` (log lift) and `13.72` (relation count). The
  proposal requires `|r| ≤ 0.3` and `VIF < 2`, so the gate fails as written.
- **Why pooled is not the right statistic:** within-rule `log m̄` range is
  `2.2e−16`, so `β_m` is exactly collinear with the rule intercept. The pooled
  correlation is between-rule heterogeneity, not a within-rule decorrelation
  failure.
- **Status:** `design_diagnostics` and `design_strata` report both levels;
  neither resolves the identification problem.
- **Recommendation:** Define the gate on an identified within-rule quantity,
  drop `β_m` or vary marginals within rule, and report a variance decomposition
  and power calculation for `β_λ`/`β_λR`.

### F. Checker Attrition — *partly resolved*

- **Observation:** The checker hard-codes `count == 3`, width `12`, height
  `11/12`, fill ratios, and palette purity.
- **Verified values:** at blur radius `1.0`, rejection is `169/200` (`84.5%`)
  and `90/100` (`90%`). Among accepted images there are zero false accepts and
  zero false rejects, so the failure mode is **attrition**, not
  misclassification.
- **Residual risk:** Conditioning on accepted images changes the estimand to a
  selected subpopulation, and rejection may differ by rule.
- **Status:** `checker.audit` and `checker_perturb.audit` now report rejection,
  coverage, and conditional/unconditional sensitivity and specificity per rule.
  Validation on generated images and human labels is **not** done.
- **Recommendation:** Replace or augment the geometric checker; report per-rule
  confusion matrices on generated images; correct RSR with a measurement-error
  or latent-class model.

### G. Statistical Model Over-Parameterization — *open*

- **Observation:** Within a rule the design fixes singleton marginals, so
  `log m̄` has no within-rule variation. The rule intercept `a_r` absorbs the
  rule mean, leaving `β_λ` identified from a narrow realized range and `β_λR`
  with little support.
- **Status:** No measurement-error or beta-binomial model is implemented.
- **Recommendation:** Use a hierarchical/measurement-error likelihood, report
  `β_λ`/`β_λR` with uncertainty under naive and corrected models, and
  pre-register `δ` separately per estimand.

---

## 2. Code & Implementation Risks

### A. Evaluation Mode and Dropout Handling — *open*

- **Files:** `exposure/pilot_train.py`, `exposure/velocity_online.py`,
  `exposure/high_t_finetune.py`, `exposure/identity_eps_finetune.py`.
- **Risk:** Dropout/EMA and inference contexts should be audited to ensure
  deterministic evaluation paths consistently suppress stochastic dropout.
- **Status:** PyTorch tests are skipped in this environment, so this was not
  executed here.

### B. Optimizer Parameter Coverage — *open*

- **Files:** `exposure/pilot_train.py`, `exposure/velocity_online.py`,
  `exposure/optimizer_probe.py`.
- **Risk:** Optimizers are built from `model.parameters()` without parameter
  groups. If weight decay is later restricted, or sub-blocks are frozen, audit
  coverage.

### C. Attention Positional Encoding Coverage — *open*

- **Files:** `exposure/pilot_train.py`, `exposure/time_embedding_probe.py`.
- **Risk:** Cross-attention runs at a `16×16` bottleneck with no explicit 2D
  spatial positional encoding. Verify whether spatial relations rely entirely
  on convolutional receptive fields.

### D. Rule Validation — *resolved*

- **Files:** `exposure/rules.py`, `exposure/render.py`, `exposure/measure.py`.
- **Status:** `validate_rule` rejects duplicate identical predicates,
  contradictory attribute values on the same variable, and cyclic strict
  relations. It is applied by `parse_rule`, so `load_rules` inherits it.
- **Residual risk:** The validator checks structural contradictions only; it
  does not detect rules that are feasible but vanishingly rare under rejection
  sampling.

### E. Uncertainty Propagation — *open*

- **Files:** `exposure/design.py`, `exposure/mixture.py`,
  `exposure/design_bounds.py`, `exposure/design_chains.py`.
- **Risk:** All LP and lift computations use the estimated A-matrix as exact.
  Pointwise Wilson intervals are not propagated to mixtures, LP feasibility, or
  lift intervals.
- **Status:** Exact LP bounds and chain diagnostics exist, but neither is a
  bootstrap/Bayesian uncertainty analysis.
- **Recommendation:** Bootstrap the A-matrix, solve robust/chance-constrained
  designs, and report lift intervals that include matrix uncertainty.

---

## 3. Priority Summary

| Priority | Risk | Blocking? | Code mitigation |
| --- | --- | --- | --- |
| P0 | Relational saturation and endpoint confound | Yes | `benchmark_audit.py` detects; benchmark not fixed |
| P0 | No learnable model / no learnability gate | Yes | none |
| P0 | Decorrelation gate fails; `β_m` collinear | Yes | diagnostics only |
| P1 | Sampled dose narrow; sampler misses feasible corner | Yes for H2 | exact LP + chain diagnostics; not wired in |
| P1 | Checker attrition on perturbed/generated images | Yes for RSR | per-rule metrics added; no generated-image validation |
| P1 | Template imbalance and missing held-out split | Yes for H3 | `benchmark_audit.py` detects; not fixed |
| P1 | A-matrix uncertainty not propagated | No, but needed | none |
| P2 | Dropout/optimizer/positional-encoding audits | No | not executed here |

---

## 4. Retracted or Corrected Claims

1. `Γ` is negative under the multiplicative null — **false**; the stated
   estimand is zero. The original table evaluated `Σ logit(c_p)`.
2. Relation saturation creates an "enormous negative `Γ`" — **false** under the
   stated formula; the problem is benchmark validity and logit boundary
   conditioning.
3. The dose is one-sided *by construction* — **not established**; the exact
   polytope admits positive lift for `23/57` rules, but the sampler does not
   reach it.
4. Within-rule `r ≈ 0` proves decorrelation — **misleading**; within-rule
   marginal variance is exactly zero, so the correlation is undefined.
5. `SE(log n) = 0.44` is the design-level uncertainty — **single-cell only**;
   mixture `SE ≈ 0.02–0.06`.
6. "`9/12` color–shape pairs" is the coverage gap — **true but weak**; only
   `3/12` are co-bound to the same object.
