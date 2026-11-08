# Verification of the review's concerns

**Date:** 2026-11-04  
**Repository commit:** `2c4db8a`  
**Purpose:** Re-derive the disputed algebra, independently verify each remaining
concern against the code and saved outputs, and separate what is proved from
what is merely asserted. This document supersedes the relevant sections of
`docs/review-2026-11-04.md`; the review is left intact as a historical record
with the author's correction notices.

**Evidence labels used below**

| Label | Meaning |
| --- | --- |
| **Proved** | Follows from the code or from an explicit derivation. |
| **Measured** | Reproduced from a saved output or a seeded simulation in this repository. |
| **Documented** | Stated in the repository's own docs, with the caveat that it is a visual or exploratory observation. |
| **Retracted** | The original review's claim does not hold and is withdrawn. |
| **Not established** | The evidence does not support the claim either way. |

---

## 0. Verdict table

| # | Claim in the review | Status | Basis |
| --- | --- | --- | --- |
| 1 | `Γ` is negative under the multiplicative null; H1 is an arithmetic identity | **Retracted** | The stated estimand is exactly zero under the null (§1). The review computed `Σ logit(c_p)`, not `logit(Π c_p)`. |
| 2 | Relation predicates are near-tautologies under existential semantics | **Proved / Measured** | Analytic `P = 0.999271`; simulation `0.99930`; saved matrix min `0.99`, median `1.00` (§2). |
| 3 | The relational contrast is confounded by the template set | **Proved** | Relation kind is assigned deterministically by bundle index; in `k_attr=2, k_rel=1`, `left_of` is subject-only and `above` is both-endpoints-bound (§3). |
| 4 | The exposure dose is one-sided and ~0.1 log units wide | **Partially retracted / re-scoped** | The 20 sampled designs are one-sided and narrow. The exact feasible polytope has median width `1.172` and admits positive lift for `23/57` rules, but the positive-lift corner is extreme and no sampled chain reaches it (§4). The defect is sampler under-dispersion, not an infeasible design space. |
| 5 | The decorrelation gate fails, and the pooled statistic is a between-rule artifact | **Proved / Measured** | Pooled `r = -0.446`, VIF `11.61`; within-rule marginal variance is exactly zero, and `98.7%` of lift variance is between-rule (§5). |
| 6 | The checker is effectively an inverse renderer and is invalid for generated images | **Proved / Measured** | Hard-coded `count==3`, width `12`, heights `11/12`, fill ratios; `84.5–90%` rejection at blur `1.0`; errors are attrition, not misclassification (§6). |
| 7 | There is no learnable model and no learnability gate | **Documented** | Stage 0 is "Not passed"; 12k samples are noise, 32k is two visual samples; the Stage 0 gate has no model criterion (§7). |
| 8 | The A-matrix and design do not propagate uncertainty | **Measured / re-scoped** | Single-cell `SE(log n)=0.44` at `n=.05`; design-level mixture `SE≈0.02–0.06`; holdout constraint violation reaches `0.0177` (§8). |
| 9 | min-KL is not the research objective; SLSQP starts at an LP vertex | **Proved** | `KL(w‖uniform)` is uniquely minimized at `w=uniform`; the saved min-KL output is uniform with `KL=4.95e-12` (§4.4). |
| 10 | `β_m` is identified only from rounding noise | **Proved / sharpened** | `log m̄` is exactly constant within rule, so it is collinear with the rule intercept and is not identified at all (§5.3). |
| 11 | `load_rules` accepts duplicate and contradictory predicates | **Fixed** | `validate_rule` now rejects duplicate predicates, contradictory same-variable attributes, and strict cycles; `parse_rule`/`load_rules` inherit it (§9). |

---

## 1. Erratum: the composition gap `Γ`

### 1.1 What the proposal says

Proposal §4.2:

> Let `c_p` be the rule-satisfaction rate (RSR) of the single-predicate rule `{p}`.
> The null is `ĉ_r = Π_{p∈P_r} c_p`. The composition gap is
> `Γ_r = logit(RSR_r) − logit(ĉ_r)`.

This is unambiguous. The second term is the logit **of the product**, not the
sum of the logits.

### 1.2 Proof that the stated estimand is zero under the null

`logit(x) = log(x/(1−x))` is strictly increasing and injective on `(0,1)`.
If `RSR_r = ĉ_r`, then

```
Γ_r = logit(RSR_r) − logit(ĉ_r) = logit(ĉ_r) − logit(ĉ_r) = 0.
```

Equivalently, `Γ_r` is the log odds ratio

```
Γ_r = log( [RSR_r/(1−RSR_r)] / [ĉ_r/(1−ĉ_r)] ),
```

which is `log(1) = 0` when the two probabilities are equal. The result holds for
every `c_p ∈ (0,1)` and every `k`. **QED.**

### 1.3 What the review actually computed

The review replaced `logit(Π c_p)` with `Σ logit(c_p)`. These are different
functions:

```
logit(Π c_p) = log( Π c_p / (1 − Π c_p) )
Σ logit(c_p) = log( Π [c_p/(1−c_p)] )
```

Their difference is `log[(1−Π c_p) / Π(1−c_p)]`, which is not zero in general.
For `c=0.8, k=2`:

```
logit(0.64)          =  0.5754
2 · logit(0.8)       =  2.7726
difference           = −2.1972   ← the review's "−2.20"
logit(0.64) − logit(0.64) = 0    ← the proposal's Γ
```

The review's table (`−2.20, −4.11, −2.95, −5.60`) is therefore a table for a
different estimand. It does not describe the proposal.

### 1.4 What survives, and what does not

**Retracted:**

- "`Γ` is not zero under the multiplicative null."
- "H1 will confirm a composition cost even for a perfectly multiplicative model."
- "The predicted increase with `K_rel` is mechanically guaranteed by the scale."
- "A singleton rate near 1 sends `logit(c_p) → +∞`, so any relational rule
  receives an enormous negative `Γ` for purely arithmetic reasons."

The last point is wrong twice over. First, the proposal uses `logit(Π c_p)`, not
`Σ logit(c_p)`. Second, the product is dominated by the attribute marginals:
adding a relation with `c≈0.999` changes `Π c_p` from `0.6240` to `0.6237`, and
changes `logit(Π c_p)` by about `0.002` (measured, §2.4). A saturated relation
does not inflate the baseline.

**Still valid as a design choice, not as an algebra defect:**

- `Γ_logit` is an odds ratio. It is zero under the null, but it is not additive
  across predicates. The log-scale gap

  ```
  Γ_log = log RSR_r − Σ_p log c_p
  ```

  is additive: each predicate contributes `log c_p`, and a perfectly
  multiplicative model contributes zero per predicate. For comparing effect
  sizes across rules with different `k` and different base rates, `Γ_log` is the
  better scale. Near `RSR≈0` or `1`, the logit is also more sensitive to small
  absolute changes.
- This is a pre-registration and interpretability issue. It is not a reason to
  declare H1 unfalsifiable. Under the stated `Γ`, H1 is falsifiable: a perfectly
  multiplicative model gives `Γ=0`, which is inside any positive equivalence
  margin `δ` and therefore *falsifies* "`Γ<0`".

**Recommendation:** make `Γ_log` primary, report `Γ_logit` as a sensitivity
analysis, and include a null simulation showing both are zero under
`RSR=Π c_p`. Do not describe `Γ_logit` as an additive composition cost.

---

## 2. Relation predicates are near-tautologies

### 2.1 Code-level semantics

`exposure/rules.py::satisfies` collects the variables named by the predicates,
iterates over permutations of distinct scene objects, and checks every predicate
under that assignment. `left_of` is `subject.x < object.x`; `above` is
`subject.y < object.y`; ties fail. Nothing requires the relation variables to be
bound to attribute-bearing objects: the existential quantifier is free to pick
any distinct pair.

So `left_of(a,b)` is true whenever the scene contains **some** ordered pair with
different `x` coordinates. With three objects, it fails only when all three
`x` coordinates are equal.

### 2.2 Analytic probability in the actual renderer

`exposure/render.py` samples three distinct cells from a `3×3` grid with centers
`{12, 32, 52}` on each axis, then adds an independent integer jitter in
`[-3, 3]` to each coordinate.

The `x` ranges of the three columns are disjoint:

```
column 0: [ 9, 15]
column 1: [29, 35]
column 2: [49, 55]
```

Therefore two objects in different columns always have different `x`. The only
way all three `x` are equal is:

1. all three cells lie in the same column, and
2. all three jitters are equal.

The probabilities are:

```
P(all three cells in one column) = 3 / C(9,3) = 3/84 = 1/28
P(all three jitters equal | same column) = 7 · (1/7)^3 = 1/49
P(all x equal) = (1/28)(1/49) = 1/1372 ≈ 0.000729
P(left_of(a,b)) = 1 − 1/1372 ≈ 0.999271
```

`above(a,b)` has the same value by symmetry. **Proved.**

### 2.3 Measurements

- Seeded base-distribution simulation, `200,000` scenes:

  | rule | rate |
  | --- | --- |
  | `left_of(a,b)` | `0.99930` |
  | `above(a,b)` | `0.99916` |
  | `red(a)` | `0.57956` |
  | `red(a) & left_of(a,b)` | `0.42907` |
  | `red(a) & square(b) & left_of(a,b)` | `0.19609` |
  | `left_of(a,b) & above(a,b)` | `0.81794` |
  | `left_of(a,b) & left_of(b,c)` | `0.89231` |

- Saved pilot matrix (`outputs/exposure-57-pilot.json`, 100 scenes/condition):
  across `3,249` relation-singleton measurements, the minimum is `0.99`, the
  median is `1.00`, and the maximum is `1.00`; `3,078` are exactly `1.0` and
  `171` are `0.99`.

- The uniform-reference relation exposure is `0.99947` (from
  `outputs/design-subspace-57.json`).

### 2.4 What this does and does not imply

**Does:**

- A model that outputs any three objects with distinct `x` (or `y`) satisfies a
  relation singleton with probability `≈0.999`. A relation singleton therefore
  cannot distinguish relational competence from arbitrary object placement.
- In a conjunction, the relation is a weak constraint on the attribute-bound
  subject. In the base distribution, `P(red(a))=0.580` falls to
  `P(red(a) & left_of(a,b))=0.429`; the relation mainly excludes the case where
  the only red object is the rightmost object. It does not test a relation
  between two *specified* objects.
- A relation with both endpoints bound is far more informative:
  `P(red(a) & square(b) & left_of(a,b))=0.196`, which is in a usable range.

**Does not:**

- It does not create an arithmetic `Γ` bias under the corrected formula. Adding
  a `0.999` relation to a rule changes the product of marginals by about
  `0.0003` and `logit(Π c_p)` by about `0.002` (measured above). The saturation
  problem is **benchmark validity**, not the scale of `Γ`.

### 2.5 Fix

Bind both endpoints and target a non-saturated marginal, e.g.
`red(a) & square(b) & left_of(a,b)`, or a relation whose truth value depends on
the two attribute-bound objects. A practical rule: reject any relation singleton
whose estimated exposure exceeds `0.9`, and require the conditional exposure
given the two bound endpoints to be nontrivial. The current 57 templates do not
do this consistently (§3).

---

## 3. The relational contrast is confounded by the template set

### 3.1 The assignment is deterministic

`exposure/templates.py::_specifications` builds each `k_rel=1` cell by taking
the bundle list and assigning:

```
bundles[:3]  → left_of
bundles[3:]  → above
```

The bundles are not exchangeable. They are ordered so that the first three have
a systematically different structure from the rest:

| cell | `bundles[:3]` (→ `left_of`) | `bundles[3:]` (→ `above`) |
| --- | --- | --- |
| `k_attr=1` | red, blue, green (colors) | yellow, circle, square, triangle (yellow + shapes) |
| `k_attr=2` | red+circle, blue+square, green+triangle on one object | yellow(a)+circle(b), square(a)+red(b), triangle(a)+circle(b) |
| `k_attr=3` | two attributes on `a`, one on `b` | one attribute on each of `a`, `b`, `c` |

So within each cell, relation kind is perfectly confounded with:

- the attribute field (`color` vs `shape`),
- whether both endpoints are attribute-bound,
- the distribution of attributes across variables.

### 3.2 Endpoint-binding contingency

Classifying every relation occurrence by whether its subject and object are
attribute-bound:

| kind | both | subject only | object only | neither |
| --- | ---: | ---: | ---: | ---: |
| `left_of` | 11 | 13 | 1 | 3 |
| `above` | 13 | 11 | 1 | 4 |

The aggregate looks balanced, but the cells are not. In `k_attr=2, k_rel=1`:

- `left_of`: `3/3` are **subject-only** (the second endpoint is free);
- `above`: `3/3` are **both-bound**.

Therefore the `left_of` vs `above` contrast in that cell is exactly the
"one endpoint vs two endpoints bound" contrast. In `k_attr=1, k_rel=1`,
`left_of` is paired only with colors and `above` only with yellow/shapes. In
`k_attr=3, k_rel=1`, `left_of` has two attributes on `a` and one on `b`, while
`above` has one on each of `a`, `b`, `c`. **Proved from the source.**

### 3.3 Coverage holes

- `9/12` color–shape pairs occur *somewhere* in the 57 templates.
- Only **`3/12`** are ever co-bound to the same object:
  `(red,circle)`, `(blue,square)`, `(green,triangle)`, each exactly `6` times.
- The other nine pairs never occur on a single object. The model is never
  trained on, for example, a red square as one object.
- There is no held-out template split. The same 57 templates are used for
  training and evaluation (the repo says so explicitly).

### 3.4 Consequence

Any relational contrast (H1's `K_rel` trend, H2's `β_λR`) is confounded with
bundle structure, endpoint binding, attribute field, and template identity.
The interaction cannot be interpreted as a relational effect. Counterbalancing
or full crossing is required, plus a held-out split.

---

## 4. Exposure variation: the sample is narrow, the feasible set is not

### 4.1 Observed in the 20 sampled designs

From `outputs/design-samples-57.json` and `outputs/exposure-57-pilot.json`:

- `1,140` rule-by-design rows.
- Every log lift is `≤ 0`: maximum `0.0`, minimum `−2.0517`, median `−0.5836`.
- Per-rule range across the 20 designs: median `0.0955`, mean `0.1400`,
  maximum `0.6314`; `44/57` rules have a range below `0.2`.
- Per-cell means:

  | cell | mean lift |
  | --- | ---: |
  | `(1,0)` | `0.000` |
  | `(1,1)` | `−0.285` |
  | `(1,2)` | `−0.744` |
  | `(2,0)` | `−0.231` |
  | `(2,1)` | `−0.623` |
  | `(2,2)` | `−1.189` |
  | `(3,0)` | `−0.502` |
  | `(3,1)` | `−1.018` |
  | `(3,2)` | `−1.591` |

- Variance decomposition: total `Var(log lift)=0.2457`, within-rule `0.0032`,
  between-rule `0.2425` (`98.7%` between-rule).

### 4.2 Exact feasible bounds (new evidence)

The 20 designs are only samples. To separate "the design space is narrow" from
"the sampler did not explore it", solve the exact linear programs

```
minimize / maximize  a_r · w
subject to           A w = A w_ref,   Σ w = 1,   w ≥ 0
```

where `A` contains the simplex row and the nine unique singleton-event rows,
`w_ref` is uniform, and `a_r` is the rule's joint-exposure row. This is the same
estimated matrix the sampler uses.

Results on the estimated matrix:

- Feasible lift interval per rule: minimum lower bound `−4.888`, maximum upper
  bound `+0.337`.
- Interval width: median `1.172`, mean `1.493`, maximum `4.951`.
- `23/57` rules can achieve positive lift; `16/57` exceed `+0.1`;
  `0/57` exceed `+0.5`.
- `34/57` rules cannot achieve positive lift, but their feasible width is still
  median `0.793`.

### 4.3 What this means

The sampled 20-design set has a median within-rule range of `0.0955`, while the
exact feasible width is `1.172`—roughly `12×` larger. `design_sample.py` runs
hit-and-run with `burn_in=100`, `thin=10`, `20` samples in a `47`-dimensional
nullspace, and its docstring says it "does not establish mixing."

To test whether this is just a small-sample artifact, I ran five independent
chains with `200` samples each, `burn_in=1000`, `thin=10`:

| chain | median per-rule range | max per-rule range | positive lifts |
| --- | ---: | ---: | ---: |
| seed 0 | `0.148` | `0.840` | `0/11,400` |
| seed 1 | `0.177` | `1.093` | `0/11,400` |
| seed 2 | `0.167` | `0.902` | `0/11,400` |
| seed 3 | `0.170` | `0.712` | `0/11,400` |
| seed 4 | `0.182` | `1.047` | `0/11,400` |

Longer chains roughly double the median range (`0.0955 → 0.15–0.18`), which is
consistent with slow mixing and insufficient burn-in, but they still do not
reach the exact width (`1.172`) and they **never** produce a positive lift in
`57,000` rule-design cells.

The positive-lift region is real but extreme. For the rule with the largest
feasible lift (`attr3_rel0_attributes_3`, `+0.337`), the LP optimum has:

- maximum weight `0.282` (about `16×` uniform);
- `47/57` weights at the floor (`≤10⁻⁶`);
- only `8` weights `≥0.01`.

So the feasible set is two-sided, but the positive side is a small corner that
requires a highly concentrated design. A hit-and-run chain initialized at the
uniform reference does not reach it, and min-KL would not either.

Therefore:

- **Retracted:** "the dose is one-sided by construction" or "the design space
  cannot support a two-sided dose." The feasible set is two-sided.
- **Verified:** the current pipeline is one-sided in practice. The 20-design
  sample and even 200-sample chains never produce a positive lift; the
  positive-lift corner is extreme and is not reached by the current sampler or
  objective.
- **Verified:** the current sample cannot identify `β_λ` or `β_λR`; the
  within-rule variation is too small relative to the feasible range.
- **Verified:** even the exact feasible set is asymmetric. Positive lift is
  capped at `+0.337`, and `34/57` rules are one-sided. A high-exposure arm is
  unavailable for the majority of rules; the experiment can at best estimate a
  one-sided slope on a subset.

### 4.4 min-KL is a no-op at the uniform reference

`design.py` implements `KL(w ‖ uniform) = Σ w log(N w)` with `N` conditions.
On the simplex:

```
d/dw [Σ w log(N w)] = log(N w) + 1 = 0  →  w = 1/N
d²/dw² = 1/w > 0
```

so the unique global minimum is `w = uniform`. Uniform satisfies the
singleton-equality constraints whenever they are set to the reference
marginals—which is exactly what `design_subspace.py` and `design_sample.py`
do. Hence the constrained min-KL solution is the uniform mixture, which by
construction applies no exposure manipulation.

The saved `outputs/design-min-kl.json` confirms this on a four-rule example:
weights `0.2499995–0.2500014`, `kl_to_uniform = 4.95e-12`. **Proved and
measured.**

The research objective is not "closest to uniform"; it is to estimate `β_λ` and
`β_λR` efficiently. A D-optimal or c-optimal design for those coefficients, or
an explicit target lift vector, is required.

---

## 5. The decorrelation gate and the between-rule artifact

### 5.1 The gate fails as written

Proposal §6 / Stage 0: "The experiment proceeds only if lift is decorrelated
from marginal exposure (`|r| ≤ 0.3, VIF < 2`)."

The repository's own diagnostics:

| matrix | pooled `r(log m̄, log lift)` | VIF `log lift` | VIF `k_rel` |
| --- | ---: | ---: | ---: |
| pilot (`design-diagnostics-57.json`) | `−0.4461` | `11.61` | `13.72` |
| holdout (`design-diagnostics-57-holdout-76001.json`) | `−0.4640` | `11.07` | `13.71` |

Both the correlation and the VIF fail the stated thresholds. The repository
labels the analysis `not_assessed`, but the proposal's criterion is explicit.
Either the gate is applied (it fails) or Stage 0 is incomplete.

### 5.2 The pooled statistic is a between-rule artifact (proof)

The design constraints fix each singleton event marginal. For any design `w`
with `A_p·w = A_p·w_ref`, the mixture marginal of event `p` is constant. The
mean marginal of a rule is the average of its predicates' fixed marginals, so
`log m̄_{r,k}` is constant in `k` for each rule `r`.

Numerically, across the 20 designs, the maximum within-rule range of
`log m̄` is `2.2×10⁻16`—machine precision. Therefore:

```
Var(log m̄) = between-rule variance only
Cov(log m̄, log lift) = Cov(E[log m̄ | r], E[log lift | r]) + 0
```

The measured decomposition is total `Var(log m̄)=0.0139`, within `0.0`,
between `0.0139`; total `Var(log lift)=0.2457`, within `0.0032`,
between `0.2425` (`98.7%`). The pooled `r=−0.446` is a between-rule
correlation driven by the fact that complex rules have both lower mean marginal
and lower lift.

On the pilot matrix the within-rule correlation is undefined (the centered
marginal column has zero variance); on the holdout matrix, where the
constraints are only approximately preserved, it is `−0.00092`.

### 5.3 Identification consequence

The primary model is

```
logit p_{r,k} = a_r + β_λ λ_{r,k} + β_λR (λ_{r,k} × K_rel,r)
               + β_m log m̄_{r,k} + ... + u_r + v_k + w_{rk}.
```

Because `log m̄_{r,k}` is constant in `k`, it is a rule-level variable and lies
in the span of the rule intercept `a_r` and the rule random effect `u_r`.
`β_m` is therefore **exactly collinear and not identified**—not merely
"identified from rounding noise." On the holdout matrix, it is identified only
from constraint violations of at most `0.0177` in probability, which are
measurement artifacts (§8).

`β_λ` is identified from within-rule variation in `λ`. As sampled, that range
is median `0.0955`; the exact feasible range is `1.172`. So it is weakly
identified by the current sample, but the design space could support it.

`β_λR` is a cross-level interaction with `K_rel` (rule-level). It is identified
from between-rule differences in the within-rule slope, which are confounded
with template identity and bundle structure (§3). It is weakly identified at
best.

---

## 6. The checker is an inverse renderer

### 6.1 Code-level proof

`exposure/checker.py::_detect_palette` requires:

- a `64×64` RGB image;
- exactly `3` connected components (`count != 3` → reject);
- every component width exactly `12`;
- every component height `11` or `12`;
- fill ratio in `[0.49, 0.56]` for height `11` (triangle),
  `[0.76, 0.84]` for height `12` (circle), or `≥0.97` for height `12` (square);
- at least `95%` of the component's pixels nearest one palette color.

These are the renderer's geometry exactly: `RADIUS=6`, `SCALE=4` gives width
`12`; the triangle spans height `11`; the fill ratios are `≈π/4`, `1`, and
`≈1/2`. The checker is the inverse of `exposure/render.py`, not a general scene
parser.

### 6.2 Measured rejection rates

`outputs/checker-perturb-41000.json` (200 scenes) and
`outputs/checker-perturb-holdout.json` (100 scenes):

| perturbation | pilot rejected | holdout rejected |
| --- | ---: | ---: |
| clean | `0/200` (`0%`) | `0/100` (`0%`) |
| brightness `0.90` | `0/200` (`0%`) | `0/100` (`0%`) |
| brightness `1.10` | `86/200` (`43.0%`) | `48/100` (`48%`) |
| blur `0.50` | `86/200` (`43.0%`) | `64/100` (`64%`) |
| blur `1.00` | `169/200` (`84.5%`) | `90/100` (`90%`) |

Among accepted images, false accepts and false rejects are `0`: the checker is
conservative, not noisy. Its failure mode is **attrition**, not
misclassification.

`checker.audit` excludes detector failures from the rule denominators
(`continue` before incrementing `positive`/`negative`). The reported `0%` rule
error is therefore conditional on acceptance. It does not bound the error on the
full distribution, and it cannot be compared to the proposal's `≤2%` gate
without specifying the target distribution.

### 6.3 Consequence

The documented model samples are blurry, noisy, or merged (12k: noise; 32k: a
red shape and a relational sample that "did not clearly satisfy" the relation).
On such images the checker will reject a large, condition-dependent fraction.
RSR is then estimated on a selected subpopulation; differential rejection by
condition biases the exposure comparison, and the human audit is blocked. The
`checker error ≤ 2%` Stage 0 gate is undefined on this evidence: `0%` on clean
accepted scenes, `43–90%` rejection on perturbed scenes.

**Fix:** validate on the actual model's generated images, report rejection
rates by condition, treat rejection as a competing outcome, and use a robust
detector or human labels with measured sensitivity and specificity. Do not use
renderer geometry as the acceptance criterion for generated images.

---

## 7. Learnability and the Stage 0 gate

### 7.1 Documented evidence

`docs/stage0-status.md` begins: **"Not passed."** The relevant entries:

- `4,000`-update fixed-scene sample: image MSE `0.8017` on `[0,1]`, "mostly
  dark noise."
- `8,000` updates: "colored blobs without clear shapes or reliable relations."
- `12,000` updates: "One red and one relational sample remained noise; two
  samples are not a quality score."
- `32,000` updates: "In two matched samples the red condition showed a red
  shape; the relational sample did not clearly satisfy the requested relation.
  These are visual observations, not an accuracy estimate."
- No Stage 1 pilot; no reference RSR; the proposed calibration target (median
  reference RSR in `[0.3,0.7]`) has never been approached.
- `docs/mosaic-replication.md`: "**no MOSAIC experiment or replication has been
  run**."

The Stage 0 gate in the proposal is: checker error `≤2%`, the design polytope
meets the decorrelation criterion, and the MOSAIC replication matches the
direction. It contains **no model learnability criterion**.

### 7.2 Consequence

Every downstream quantity—RSR, `N50`, `Γ`, lift, allocation—is a function of a
model that has not demonstrated the benchmark. Measuring now measures the
model's failure to learn, not the exposure effect. The absence of a learnability
gate means Stage 0 can be "passed" on instrument checks while the model is
unusable.

**Fix:** add a Stage 0 gate such as: a reference model reaches median RSR in
`[0.3,0.7]` on held-out rules, with a binomial confidence interval, before any
exposure experiment. The 32k result is a visual observation of two samples, not
a quantitative RSR; it should be reported as such.

---

## 8. Measurement error and design uncertainty

### 8.1 Single-cell versus design-level uncertainty

For a binomial proportion `n` estimated from `N` scenes,

```
SE(log n) = sqrt((1−n) / (n N)).
```

At `n=0.05, N=100`, this is `0.436`; at `N=10^5`, `0.0138`. The review's
`0.44` is a **single-cell** figure.

The design lift is a mixture over columns: `p = Σ_c w_c p_c`, with

```
Var(p) = Σ_c w_c² p_c (1−p_c) / N_c,
SE(log p) ≈ sqrt(Var(p)) / p.
```

On the saved 20 designs, the median `SE(log joint)` is `0.0226`; the rarest
complex rules reach about `0.06`. So the design-level uncertainty is
`0.02–0.06`, not `0.44`. It is still material relative to the median
within-rule range `0.0955`: for `12/57` rules the observed range is less than
twice their own median standard error.

### 8.2 The equality constraints are not exact

On the training matrix, the singleton constraints hold to `1.89×10⁻16` by
construction. On an independent holdout matrix, the same fixed designs violate
them by up to **`0.017673`** in probability
(`outputs/design-holdout-76001.json`). So "singleton-preserving" is a property
of the estimated matrix, not of the population. The LP and sampler treat the
constraints as exact; on the truth, some designs may be infeasible and the
achieved lift differs.

### 8.3 Design objective and optimizer

- min-KL is a no-op at uniform (§4.4).
- SLSQP starts from the LP vertex (`solution.x` from `linprog`), a corner where
  the KL gradient is steep; this is a poor initialization and can converge to a
  boundary solution.
- The objective is not the research objective: it does not maximize information
  about `β_λ` or `β_λR`. A D-optimal or c-optimal design for those coefficients
  is appropriate, with bootstrap or Bayesian propagation of the A-matrix
  uncertainty.

---

## 9. Template coverage and rule validation

- `57` templates across `9` cells, as documented.
- `9/12` color–shape pairs occur somewhere; only `3/12` are co-bound to the
  same object (§3.3).
- No held-out template split; the templates are both training and evaluation.
- `load_rules` now rejects duplicate identical predicates, contradictory
  predicates, and cyclic strict relations through `validate_rule`:
  - duplicate identical predicates were previously semantically redundant but
    doubled the predicate count, distorting `k_attr`/`k_rel` and the mean
    marginal;
  - contradictory predicates (e.g. `red(a) & blue(a)`) are unsatisfiable; as a
    *condition*, `measure.estimate` would otherwise exhaust `max_attempts` and
    raise `ValueError` after `max_attempts` scenes;
  - a strict cycle (e.g. `a left_of b & b left_of a`) is unsatisfiable.
- `canonical_key` still deduplicates for template validation; the measurement
  path now rejects duplicates directly.

---

## 10. What survives

The following concerns survive independent verification and remain blocking:

1. **Relational saturation and the template confound.** Relation singletons are
   `≈0.999`; the relational contrast is confounded with endpoint binding,
   attribute field, and bundle structure. H1's `K_rel` trend and H2's `β_λR` are
   not identified as relational effects. (§2–§3)
2. **No learnable model, no learnability gate.** Stage 0 is not passed; the
   model's samples are documented as noise/blobs; the gate does not test the
   model. No downstream outcome is measurable. (§7)
3. **The decorrelation gate fails as written, and the pooled statistic is a
   between-rule complexity artifact.** `β_m` is exactly collinear with the rule
   intercept. (§5)
4. **The checker is an inverse renderer.** On generated images it will reject
   differentially; RSR is then a selected-subpopulation estimate. (§6)
5. **The current 20-design sample is under-dispersed.** `β_λ` and `β_λR` are
   weakly identified as sampled, although the exact feasible set is wider.
   (§4)
6. **min-KL is a no-op at uniform, and the design does not target the research
   objective.** (§4.4, §8.3)

The following are retracted or corrected:

1. `Γ` is negative under the multiplicative null—**false**; the stated estimand
   is zero. (§1)
2. Relation saturation creates an "enormous negative `Γ`"—**false** under the
   stated formula; the problem is benchmark validity. (§2.4)
3. The dose is one-sided by construction—**not established**; the sample is
   under-dispersed, the feasible set admits positive lift for `23/57` rules, and
   the positive-lift region is an extreme corner that no sampled chain reaches.
   (§4)
4. Within-rule `r≈0` as evidence of decorrelation—**misleading**; the
   within-rule marginal variance is exactly zero, so the correlation is
   undefined, and the pooled `r` is between-rule. (§5)
5. `SE(log n)=0.44` as the design-level uncertainty—**single-cell only**;
   mixture `SE≈0.02–0.06`. (§8)
6. "`9/12` color–shape pairs" as the coverage gap—**true but weak**; only
   `3/12` are co-bound to the same object. (§3.3)

---

## 11. Falsifiable fixes and acceptance criteria

| # | Fix | Acceptance criterion |
| --- | --- | --- |
| F1 | Pre-register `Γ_log = log RSR − Σ log c_p` as primary; report `Γ_logit` as sensitivity; add a null simulation. | Both statistics are within Monte Carlo error of `0` under `RSR=Π c_p`. |
| F2 | Bind both relation endpoints to attributes; target relation singletons in `[0.2,0.8]`. | Every relation singleton `<0.9`; conditional exposure given both bound endpoints `>0.1`. |
| F3 | Counterbalance relation kind against bundle structure; add a held-out template split; cover all `12` same-object color–shape pairs. | Randomization test shows no association between relation kind and bundle features (`p>0.05`); holdout has at least one template per cell. |
| F4 | Replace min-KL with a D-optimal/c-optimal design for `β_λ`/`β_λR`; run multi-chain hit-and-run with R-hat/ESS; report exact LP bounds. | Realized within-rule `λ` range covers `≥50%` of the exact LP width; chain diagnostics pass. |
| F5 | Report within-rule and between-rule correlations separately; drop `β_m` or vary marginals within rule. | Design matrix has full rank for the reported coefficients; the gate is defined on an identified quantity. |
| F6 | Validate the checker on generated images; report rejection by condition; use a robust detector or human labels with sensitivity/specificity. | Rejection `<5%` and balanced across conditions; per-rule sensitivity and specificity `≥0.95`. |
| F7 | Add a Stage 0 learnability gate. | A reference model reaches median RSR in `[0.3,0.7]` on held-out rules, with a CI excluding `0`. |
| F8 | Bootstrap the A matrix and propagate to designs and lifts. | Design ranking and lift intervals are stable under bootstrap. |
| F9 | Reject duplicate, contradictory, and cyclic rules at load time. | `load_rules` and `parse_rule` reject all three classes with a clear error. |

### Implementation status (2026-11-05)

| # | Status | Evidence / remaining work |
| --- | --- | --- |
| F1 | **Implemented** | `exposure/composition.py` reports `gamma_logit`, `gamma_log`, `gamma_status`, and `marginal_product`; `exposure/mixture.py` emits them per rule and pair; `tests/test_composition.py` verifies the null and boundary behavior. The binding-aware maximum-entropy null remains open. |
| F2 | **Not implemented** | `exposure/benchmark_audit.py` detects saturation and endpoint confounding; the renderer and template semantics are unchanged. |
| F3 | **Not implemented** | `benchmark_audit.py` reports `3/12` same-object pairs and `no_held_out_template_split`; no factorial generator or holdout exists. |
| F4 | **Partial** | `exposure/design_bounds.py` computes exact LP bounds and the positive corner; `exposure/design_chains.py` reports split-R-hat and autocorrelation ESS; `tests/test_design_bounds.py` and `tests/test_design_chains.py` cover them. The D-optimal/c-optimal objective and wiring into the default sampler are missing. |
| F5 | **Partial** | `design_diagnostics` and `design_strata` document the exact within-rule collinearity (`log m̄` range `2.2e−16`); the gate is still reported as pooled by default, and no fitted model drops or reparameterizes `β_m`. |
| F6 | **Partial** | `checker.audit` and `checker_perturb.audit` now report rejection rate, coverage, and conditional/unconditional sensitivity/specificity per rule; `tests/test_checker.py` and `tests/test_checker_perturb.py` cover the fields. No generated-image or human-labelled validation exists. |
| F7 | **Not implemented** | Documented as a proposed gate in `docs/stage0-status.md` and `docs/experiment-spec.md`; no model criterion is enforced. |
| F8 | **Not implemented** | LP and lift computations still use the estimated matrix as exact. |
| F9 | **Implemented** | `exposure/rules.validate_rule` rejects duplicate predicates, contradictory same-variable attributes, and cyclic strict relations; `render.parse_rule` calls it, so `measure.load_rules` inherits it; `tests/test_rules.py` covers all three. |

---

## Appendix A. Reproduction

All commands assume the repository root and `uv run --no-sync`.

**Tests**

```sh
uv run --no-sync python -m unittest discover -s tests -q
# Ran 209 tests ... OK (skipped=55)
```

**New code-level checks**

```sh
# Corrected composition-gap null (both gaps zero under RSR = product(c_p))
uv run --no-sync python -m exposure.composition --rsr 0 --marginals 0.8 0.8 --null

# Exact LP exposure/lift bounds and positive-lift corner
uv run --no-sync python -m exposure.design_bounds \
  --report outputs/exposure-57-pilot.json \
  --weights outputs/uniform-57.json \
  --output outputs/design-bounds.json

# Multi-chain split-R-hat and autocorrelation ESS
uv run --no-sync python -m exposure.design_chains \
  --report outputs/exposure-57-pilot.json \
  --weights outputs/uniform-57.json \
  --chains 4 --samples 100 --seed 0 \
  --burn-in 1000 --thin 10 \
  --output outputs/design-chains.json

# Adversarial benchmark-validity audit (exits 2 with --require-confirmatory)
uv run --no-sync python -m exposure.benchmark_audit \
  --rules outputs/templates-57.json \
  --output outputs/benchmark-audit.json
```

**Relation saturation (analytic and simulated)**

```sh
uv run --no-sync python - <<'PY'
import math, random
from exposure.render import _sample_objects
from exposure.rules import Relation, Attribute, satisfies
p_fail = (3 / math.comb(9, 3)) * (1 / 49)
print("analytic P(left_of) =", 1 - p_fail)
def rate(preds, n=200_000, seed=0):
    rng = random.Random(seed); hits = 0
    for _ in range(n):
        hits += satisfies(_sample_objects(rng), preds)
    return hits / n
print("sim left_of =", rate((Relation("a", "left_of", "b"),)))
print("sim above   =", rate((Relation("a", "above", "b"),)))
print("sim red&left=", rate((Attribute("a","color","red"),
                             Relation("a","left_of","b"))))
print("sim both-bound red/square/left =",
      rate((Attribute("a","color","red"),
            Attribute("b","shape","square"),
            Relation("a","left_of","b"))))
PY
```

**Exact feasible lift bounds**

```sh
uv run --no-sync python - <<'PY'
import json, math, numpy as np
from scipy.optimize import linprog
from exposure.measure import load_rules, unique_object
from exposure.rules import Attribute
report = json.load(open("outputs/exposure-57-pilot.json"),
                   object_pairs_hook=unique_object)
rules = load_rules(report["rules"]); names = list(rules)
cols = {c["condition"]: c for c in report["columns"]}
def event(p):
    return (("attribute", p.field, p.value) if isinstance(p, Attribute)
            else ("relation", p.kind))
events = {}
for name, preds in rules.items():
    for i, p in enumerate(preds):
        key = event(p)
        vec = np.array([next(e["probability"] for e in
                             cols[c]["predicate_subsets"][name]
                             if e["predicate_indices"] == [i])
                        for c in names])
        events.setdefault(key, vec)
A = np.vstack([np.ones(len(names)), *[events[k] for k in sorted(events)]])
wref = np.full(len(names), 1 / len(names)); target = A @ wref
lows, highs = [], []
for name, preds in rules.items():
    a = np.array([cols[c]["measurements"][name]["probability"] for c in names])
    lo = linprog(a, A_eq=A, b_eq=target, bounds=[(0, None)] * len(names),
                 method="highs").fun
    hi = -linprog(-a, A_eq=A, b_eq=target, bounds=[(0, None)] * len(names),
                  method="highs").fun
    lp = sum(math.log(float(events[event(p)] @ wref)) for p in preds)
    lows.append(math.log(lo) - lp); highs.append(math.log(hi) - lp)
print("min lower", min(lows), "max upper", max(highs))
print("width median", np.median(np.array(highs) - np.array(lows)))
print("positive", sum(h > 1e-9 for h in highs), "/", len(highs))
PY
```

**Longer multi-seed chains**

```sh
uv run --no-sync python - <<'PY'
import json, numpy as np
from exposure.measure import load_rules, unique_object
from exposure.mixture import summarize
from exposure.design_sample import sample_designs
report = json.load(open("outputs/exposure-57-pilot.json"),
                   object_pairs_hook=unique_object)
names = list(load_rules(report["rules"]))
wref = {n: 1 / len(names) for n in names}
for seed in range(5):
    out = sample_designs(report, wref, samples=200, seed=seed,
                         min_weight=0.001, burn_in=1000, thin=10)
    per = {n: [] for n in names}
    for design in out["designs"]:
        s = summarize(report, design["weights"])
        for n in names:
            if s["results"][n]["log_lift"] is not None:
                per[n].append(s["results"][n]["log_lift"])
    ranges = [max(v) - min(v) for v in per.values()]
    vals = [x for v in per.values() for x in v]
    print(seed, "median range", np.median(ranges),
          "max", max(ranges), "positive", sum(x > 1e-9 for x in vals), "/", len(vals))
PY
```

**Checker perturbation table**

```sh
uv run --no-sync python -m exposure.checker_perturb \
  --samples 200 --seed 41000 --output outputs/checker-perturb-41000.json
uv run --no-sync python -m exposure.checker_perturb \
  --samples 100 --seed 20000 --output outputs/checker-perturb-holdout.json
```

**Existing diagnostics cited above**

```sh
# design correlations and VIF
uv run --no-sync python -m exposure.design_diagnostics \
  --report outputs/exposure-57-pilot.json \
  --designs outputs/design-samples-57.json \
  --output outputs/design-diagnostics-57.json
# within-rule vs pooled
uv run --no-sync python -m exposure.design_strata \
  --report outputs/design-diagnostics-57.json \
  --output outputs/design-strata-57.json
# holdout constraints
uv run --no-sync python -m exposure.design_holdout \
  --training outputs/exposure-57-pilot.json \
  --holdout outputs/exposure-57-holdout-76001.json \
  --reference outputs/uniform-57.json \
  --designs outputs/design-samples-57.json \
  --output outputs/design-holdout-76001.json
```

**Relevant saved outputs**

- `outputs/exposure-57-pilot.json` — 57-rule, 100-scene/condition matrix.
- `outputs/exposure-57-holdout-76001.json` — independent 100-scene matrix.
- `outputs/design-samples-57.json` — 20 hit-and-run designs.
- `outputs/design-diagnostics-57.json` — pooled correlations and VIF.
- `outputs/design-strata-57.json` — pooled vs within-rule correlations.
- `outputs/design-subspace-57.json` — singleton events and nullspace.
- `outputs/design-holdout-76001.json` — training vs holdout deviations.
- `outputs/checker-perturb-41000.json` — checker perturbation table.
- `outputs/checker-perturb-holdout.json` — checker holdout table.
- `outputs/design-min-kl.json` — min-KL returns the uniform mixture.

---

## Appendix B. One-paragraph summary

The review's central algebra claim was wrong: the proposal's
`Γ = logit(RSR) − logit(Π c_p)` is exactly zero under the multiplicative null,
and the review's `−2.20/−4.11` table belongs to `Σ logit(c_p)`, a different
estimand. That correction does not rescue the research design. Relation
predicates are near-tautologies: under the actual grid renderer and existential
semantics, `P(left_of)=0.999271` analytically and `0.99930` in simulation, so a
relation singleton cannot measure relational competence, and the templates
confound relation kind with endpoint binding and attribute structure. The
current 20-design sample is one-sided and narrow (median within-rule lift range
`0.0955`), but this is a sampler problem, not a proven design limit: exact LP
bounds give a median feasible width of `1.172` and positive lift for `23/57`
rules, although the positive-lift corner is extreme (max weight `0.282`, `47/57`
weights at the floor) and no sampled chain reaches it. min-KL is a no-op at the
uniform reference. The decorrelation gate
fails as written (`r=−0.446`, VIF `11.61`), and the pooled statistic is a
between-rule artifact because the design fixes the marginals exactly, making
`β_m` collinear with the rule intercept. The checker is the inverse renderer
and rejects `84.5–90%` of blur-`1.0` images; its errors are attrition, not
misclassification. Stage 0 is explicitly not passed, the model's samples are
documented as noise or two visual observations, and the gate has no
learnability criterion. The defensible path is to fix the scale
(pre-register `Γ_log`), bind relation endpoints, counterbalance templates with a
held-out split, replace min-KL with a coefficient-optimal design and a
converged sampler, validate the checker on generated images, and add a
learnability gate before any exposure experiment.
