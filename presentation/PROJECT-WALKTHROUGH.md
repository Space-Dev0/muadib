# Which Exposure Matters? — Complete Project Walkthrough

> Audience: someone who has never seen this repository and must present it.
> Read sections 1–5 first (about 20 minutes). Use sections 6–10 as reference.
> This guide is descriptive, not promotional: the project is an instrument-building
> effort whose Stage 0 validation has **not passed**.

---

## 0. The 10-minute path

If you only have ten minutes before presenting, read these in order:

1. §1 — the one-paragraph explanation.
2. §2 — the research question and the five quantities you must be able to define:
   exposure, marginal, lift, composition gap, N50.
3. §3 — the synthetic world (three objects, four colours, three shapes, two relations).
4. §5 — what has actually been done versus what has not.
5. §11 — the Stage 0 dashboard and the honest caveats.

Then use §13 as your spoken script and §14 for the live demo commands.

---

## 1. The project in one paragraph

Modern image generators can often draw a red object and can often draw a square,
but they fail more often when asked for *a red object to the left of a square*.
This project asks **which kind of training exposure fixes that**:

- exposure to the **individual ingredients** (red; square),
- exposure to **pairs of ingredients** (red + square, without the relation),
- exposure to the **complete rule** (red left of square), and
- exposure to **relations** (left-of, above).

The scientific move is to vary the joint/rule exposure while holding the
individual ingredient frequencies fixed. The repository builds the instruments
needed for that experiment: a controlled synthetic scene world, formal rule
semantics, a Monte Carlo exposure matrix, a linear-programming design tool,
a small conditional diffusion model, an image checker, and blinded human-audit
tooling. A 63-slide Beamer deck in `presentation/` tells the research story.

**The single most important sentence for your talk:** the repository is a
*staged instrument-development project*. No confirmatory experiment has been run,
and its own documentation says Stage 0 has not passed. Presenting it as a finished
study would be wrong; presenting it as a carefully built experimental apparatus
with honest negative/inconclusive evidence is accurate.

---

## 2. From zero: the research question

### 2.1 Conditional diffusion in one minute

- An image is a grid of numbers (here 64×64 pixels × 3 colour channels).
- A **conditional diffusion model** takes a description (a condition) and a noisy
  image, and learns to predict the noise that was added.
- Training: take a clean image `x0`, pick a timestep `t`, add Gaussian noise
  `ε` to get `xt`, and train a network `εθ(xt, t, condition)` to recover `ε`.
- Generation: start from pure noise and repeatedly subtract predicted noise.
- The condition here is not a sentence; it is a **structured rule**: attribute
  predicates over named object variables plus spatial relations.

The exact equations used in the deck:

```
Forward:      q(x_t | x_0) = N( sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I )
Training:     L = E[ || epsilon - epsilon_theta(x_t, t, c) ||^2 ]
Reverse step: x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(1 - beta_t) + noise
Guidance:     eps_tilde = eps_uncond + s * (eps_cond - eps_uncond)
```

### 2.2 Rules and the synthetic world

A **rule** is a conjunction of predicates over variables. Example
(`examples/red_left_of_square.json`):

```json
[
  {"type": "attribute", "variable": "a", "field": "color", "value": "red"},
  {"type": "attribute", "variable": "b", "field": "shape", "value": "square"},
  {"type": "relation", "subject": "a", "kind": "left_of", "object": "b"}
]
```

Meaning: there exist two **distinct** objects such that `a` is red, `b` is
square, and `a` is left of `b`. Repeated variable names must bind to the same
object; different variables must bind to different objects.

The world is deliberately tiny and countable:

| Property | Values |
|---|---|
| Image | 64×64 RGB |
| Objects per scene | exactly 3 |
| Colours | red, blue, green, yellow |
| Shapes | circle, square, triangle |
| Relations | `left_of`, `above` (strict, by object centre) |
| Placement | 3×3 grid of centres, sample 3 distinct cells, jitter ±3 px |

Because the renderer emits ground-truth annotations (colour, shape, x, y), the
project can compute exactly whether a scene satisfies a rule. That removes human
label noise and lets exposure be measured rather than guessed.

### 2.3 Exposure

For a predicate set `S`, exposure is

```
n(S) = P(scene satisfies S)
```

computed from **realized annotations**, not from which template requested the
scene. This is essential: a scene generated for "red left of square" may also
satisfy "red" and "square" incidentally. The project counts that incidental
exposure.

- **Predicate marginal**: `m_p = n({p})`.
- **Rule exposure**: `n_r = n(P_r)` for the full conjunction.
- **Independence reference**: `n_r^ind = product of m_p`.
- **Lift**: `lambda_r = log(n_r / n_r^ind)`. Negative lift = the combination is
  rarer than its ingredients would suggest (suppression). Positive lift =
  synergy.
- **Breadth**: entropy-based measure of how many different partner contexts a
  predicate appears in.

### 2.4 Why this is not just class imbalance

Suppose dataset A has red in 50% of scenes and square in 40%, and so does
dataset B. Their marginals match. But A might have red-left-of-square in 10% of
scenes while B has it in 0.1%. Class-frequency balancing cannot see this
difference; joint exposure/lift can.

### 2.5 Composition gap

Let `RSR_r` be the model's **rule satisfaction rate** for rule `r`, and `c_p`
the satisfaction rate for each single-predicate rule. The multiplicative null
(inspired by Okawa et al.) is `RSR_r = product(c_p)`. The proposal defines the
logit-scale gap:

```
Gamma_logit = logit(RSR_r) - logit(product c_p)
Gamma_log   = log RSR_r - sum log c_p        (additive sensitivity scale)
```

Both are exactly zero under the multiplicative null in the interior of (0,1).
They are different transforms, not competing nulls. Boundary values (0 or 1)
make the logit undefined, so the code returns `None` plus an explicit status
instead of inventing infinities.

There is a documented correction in this repo: an earlier review used
`logit(product c_p) - sum logit(c_p)`, which is a *different* quantity and is
not the proposal's estimand. The code implements the correct version. If you
present the gap, say "the project corrected an earlier algebra mistake".

### 2.6 N50

`N50` is the exposure (or number of training examples) at which a rule's
satisfaction crosses 50%. It measures sample efficiency without floor/ceiling
effects. The proposal distinguishes exposure probability `n50` from total
exposure count `N50 = steps × batch × n`.

### 2.7 The hypotheses

| ID | Claim | Falsified if |
|---|---|---|
| H1 | Composition costs: full-rule RSR falls below the independence baseline, more so as relational complexity `K_rel` rises | gap inside equivalence margin in all complexity cells |
| H2 | At fixed marginals, the gap increases with lift; slope larger for relational rules | slope CI inside ±delta, or slopes do not differ by `K_rel` |
| H3 | Zero-shot success on a held-out rule needs pairwise support: leaving out only the rule is survivable, leaving out its pairs is not | equal zero-shot success across leave-out types |
| H4 | Breadth matters beyond frequency: broad predicates need fewer examples to reach N50 | N50 shows no breadth dependence after controlling frequency |
| H5 | Transfer features improve out-of-run prediction of per-rule learning speed | no cross-validated gain / sign agreement at chance |
| H6 | Composition-gap loss concentrates in specific log-SNR bands and spatial regions (object gaps/boundaries for relations) | loss spread uniformly |

Stage 0 is the instrumentation gate that must pass before any of these are
tested.

---

## 3. The pipeline, end to end

```
             examples/*.json
                  |
                  v
  +-------------------------------+
  | 1. RULE SEMANTICS             |  exposure/rules.py
  |    parse + validate + satisfy |  exposure/render.py (parse_rule)
  +-------------------------------+
                  |
                  v
  +-------------------------------+     64x64 PNG + JSON annotations
  | 2. SYNTHETIC RENDERER         | --> outputs/example.png/.json
  |    grid scenes, rejection     |     exposure/render.py
  +-------------------------------+
                  |
                  v
  +-------------------------------+     schema-1 matrix with Wilson CIs
  | 3. EXPOSURE MEASUREMENT       | --> outputs/exposure-subsets.json
  |    sample per condition,      |     exposure/measure.py
  |    cross-evaluate all rules   |
  +-------------------------------+
                  |
                  v
  +-------------------------------+     weighted exposure, log lift, gaps
  | 4. MIXTURE ANALYSIS           | --> outputs/mixture.json
  |    weights over conditions    |     exposure/mixture.py + composition.py
  +-------------------------------+
                  |
                  v
  +-------------------------------+     LP feasibility / min-KL weights
  | 5. EXPOSURE DESIGN            | --> outputs/design.json
  |    simplex + linear bounds    |     exposure/design*.py
  +-------------------------------+
                  |
                  v
  +-------------------------------+     schema-v3 checkpoint
  | 6. MODEL TRAINING             | --> outputs/pilot-run-*/pilot.pt
  |    online render, DDPM, EMA   |     exposure/pilot_train.py
  +-------------------------------+
                  |
                  v
  +-------------------------------+     PNG samples
  | 7. SAMPLING                   | --> outputs/pilot-run-*/*.png
  |    DDPM reverse + guidance    |     exposure/pilot_train.py
  +-------------------------------+
                  |
                  v
  +-------------------------------+     detected objects, per-rule metrics
  | 8. IMAGE CHECKER + AUDIT      | --> outputs/checker-audit.json
  |    pixel heuristics, human    |     exposure/checker.py, human_audit.py
  |    blinded review packs       |
  +-------------------------------+
                  |
                  v
  +-------------------------------+
  | 9. STAGE 0 GATES              |  docs/stage0-status.md
  |    learnability / checker /   |  exposure/benchmark_audit.py
  |    decorrelation / MOSAIC     |  exposure/design_diagnostics.py
  +-------------------------------+
```

Steps 1–5 are CPU-only and fully implemented. Step 6 is optional (`--extra train`)
and exploratory. Steps 7–9 exist but have not produced validated research
results.

---

## 4. Repository map

```
muadib/
├── README.md                    entry point and doc index
├── pyproject.toml               deps: pillow, scipy; optional torch 2.9.1+cu128
├── examples/                    small rule/weight/constraint JSON files
├── exposure/                    all Python modules (~70 files, ~12.8k lines)
├── tests/                       ~300 test methods (training tests need torch)
├── docs/                        ~78 design/evidence documents
├── presentation/                63-slide Beamer deck + speaking guides
├── scripts/                     Windows/GPU launchers and supervisors
├── outputs/                     generated artifacts (git-ignored)
├── analysis_outputs/            prior static analysis summary, risks, contract
└── Which Exposure Matters ...   the Rev. 4 research proposal (PDF)
```

The name `muadib` is just the repository name; the Python package is `exposure`
and the project name is `which-exposure-matters`.

---

## 5. What has actually been done (evidence, not claims)

### 5.1 Timeline

| Period | Work |
|---|---|
| 2026-10-31 | Proposal added; rule evaluator, synthetic renderer, conditional sampling, Monte Carlo measurement |
| early Nov | Templates, grammar audit, composition gaps, LP design, design subspace/sampling/bounds/chains, mixture validation |
| 2026-11-04/05 | Internal code and research review; algebra correction and independent verification |
| Nov 5–8 | Checker audits, perturbation tests, human-audit plumbing, Gaussian diagnostic, MOSAIC source review |
| Nov 8–14 | GPU pilot training (epsilon, velocity, fixed-scene probes), preview reviews, 300-image restricted pilot audit, checker geometry investigation |
| Nov 14–16 | MOSAIC-inspired 2×2 matched-cell experiment: four cells, distributed 500k-update continuation; exploratory only |

### 5.2 Concrete artifacts

- **Rule engine and renderer**: working, tested; 64×64 annotated scenes.
- **Exposure measurement**: produces a condition × rule matrix with Wilson
  intervals and optional singleton/pair subsets.
- **Mixture analysis**: weighted exposure, log lift, both composition gaps.
- **Design tools**: LP feasibility, minimum-KL allocation, SVD nullspace,
  hit-and-run sampling, exact LP bounds, multi-chain R-hat/ESS.
- **57-template candidate benchmark** across 9 complexity cells.
- **PilotUNet**: a small conditional U-Net with tokenized predicate
  conditioning and cross-attention; online rendering; DDPM; EMA; classifier-free
  guidance; deterministic mode; resumable v3 checkpoints.
- **Velocity/flow-matching variants** and many diagnostic probes (denoising,
  reconstruction, high-t baseline, optimizer, time embedding, identity residual).
- **Checker**: pixel heuristic + perturbation audit + experimental narrow-shape
  variants.
- **Human audit**: blinded review-pack preparation, scoring, agreement.
- **MOSAIC-inspired 2×2 experiment**: low/high scene complexity ×
  balanced/imbalanced concept distribution, two laptops, four cells.

### 5.3 What the evidence actually says

| Claim | Status |
|---|---|
| The renderer and rule semantics work | Yes, tested. |
| Exposure can be measured and manipulated on paper | Yes, on the estimated CPU matrix. |
| A trained diffusion model reliably generates the requested scenes | **No.** Early online samples were noise; later 100k–800k previews show recognizable shapes but inconsistent object counts/relations. |
| The image checker is validated on generated images | **No.** Clean renders: good. Blur radius 1.0: 84.5–90% rejection (attrition). Generated-image audits are small and narrow. |
| Exposure can be decorrelated from complexity | **Not yet.** Pooled `r = -0.446`, VIF `11.61`, against proposal thresholds `|r| ≤ 0.3`, `VIF < 2`. |
| The 57-template benchmark is confirmatory-ready | **No.** `benchmark_audit` returns `confirmatory_ready: false`: relation singletons saturate at 0.999271, relation kind is confounded with endpoint binding, only 3/12 colour–shape pairs are co-bound, no held-out split. |
| MOSAIC has been replicated | **No.** Source review plus a small MOSAIC-inspired contrast; no completed replication result. |
| Stage 0 passed | **No. Explicitly not passed.** |
| A confirmatory H1–H6 result exists | **No. No Stage 1 pilot exists.** |

This honesty is a feature of the repository: nearly every document records its
own limitations. Use that in the talk; it makes the work credible.

---

## 6. Core code, line by line

The following are the files you must understand. Line numbers refer to the
current working tree.

### 6.1 `exposure/rules.py` (138 lines) — the formal semantics

| Lines | What happens |
|---|---|
| 13–18 | `SceneObject`: frozen dataclass with `color, shape, x, y`. Frozen makes objects hashable and prevents mutation. |
| 21–29 | `Attribute`: variable + field (`color`/`shape`) + value; rejects unsupported fields. |
| 32–42 | `Relation`: subject + kind (`left_of`/`above`) + object; rejects self-relations. |
| 45 | `Predicate = Attribute | Relation`, a type alias. |
| 48–67 | `_has_cycle`: depth-first search over a directed graph of strict orderings. A cycle such as `a left_of b`, `b left_of a` is unsatisfiable. |
| 70–102 | `validate_rule`: rejects empty rules, duplicate predicates, contradictory attribute values for the same variable (`a red` and `a blue`), and cycles. This prevents invalid rules from silently entering measurement or design. |
| 105–128 | `satisfies`: collect all variables, then try every permutation of distinct scene objects assigned to those variables. Return `True` if any assignment matches every predicate. This is existential quantification with distinct bindings. Permutations are expensive but the scene has only 3 objects. |
| 131–138 | `_matches`: attribute equality; `left_of` is `subject.x < target.x`; `above` is `subject.y < target.y`. Strict inequalities mean ties fail. Image coordinates: y grows downward. |

Key semantic point to say aloud: "a scene satisfies the rule if *some* distinct
assignment of objects does". The same scene can satisfy many rules, which is why
exposure is measured from annotations.

### 6.2 `exposure/render.py` (146 lines) — the synthetic world

| Lines | What happens |
|---|---|
| 18–27 | Palette: 4 RGB colours; shapes; 64×64 output; 4× supersampling; radius 6. |
| 30–33 | `RenderedScene`: image plus ground-truth object tuple. |
| 36–44 | `_sample_objects`: build 9 grid centres; choose 3 distinct cells; jitter each by ±3 px; choose colour and shape uniformly. |
| 47–61 | `_draw`: render at 4× (256×256), draw ellipse/rectangle/polygon, then LANCZOS-downsample to 64×64 for antialiasing. |
| 64–66 | `render_scene(seed)`: deterministic unconditional scene. |
| 69–76 | `render_conditioned`: calls rejection sampler, then draws. |
| 79–91 | `sample_conditioned`: repeatedly sample a scene until it satisfies the rule, up to `max_attempts`. Failure raises; it does **not** prove the rule is infeasible (it may just be rare). |
| 94–110 | `parse_rule`: strict JSON schema; validates attribute values against the palette and shape list; calls `validate_rule`. |
| 113–142 | CLI: `--seed`, `--output`, `--rule`, `--max-attempts`; writes PNG plus JSON with objects, rule, attempts. |

### 6.3 `exposure/measure.py` (138 lines) — the exposure matrix

| Lines | What happens |
|---|---|
| 18–25 | `unique_object`: JSON object hook that rejects duplicate keys, which would otherwise silently overwrite part of a rule. |
| 28–40 | `load_rules`: parse an ordered mapping of rule names to predicate lists; preserve order. |
| 43–50 | `wilson_interval`: 95% binomial interval, including at 0 and 1. |
| 53–111 | `estimate`: for each **condition** (column), draw `samples` accepted scenes; for every **measured rule** (row), count how many scenes satisfy it. If `include_subsets`, also count all singletons and all pairs inside each rule. Store successes, probability, Wilson interval. The row's own condition is forced to probability 1.0. |
| 107–111 | Output schema: seed, samples per condition, rule order, columns, interpretation string. |
| 114–134 | CLI: `--rules`, `--samples`, `--seed`, `--max-attempts`, `--include-subsets`, `--output`; embeds the original rule definitions in the report. |

Important nuance: columns are *requested conditions*, rows are *realized
events*. This is exactly the proposal's `A` matrix (`A[S, r] = P(scene satisfies S | scene requested for r)`).

### 6.4 `exposure/composition.py` (159 lines) — the gaps

| Lines | What happens |
|---|---|
| 44–50 | `logit(p) = log(p/(1-p))`, rejecting boundary values. |
| 53–69 | `gamma_logit`: `logit(RSR) - logit(product of marginals)`. |
| 72–79 | `gamma_log`: `log(RSR) - sum(log marginals)`, the additive scale. |
| 82–120 | `composition_gaps`: validates inputs, computes both, returns `None` plus a status when a gap is undefined (`both_defined`, `log_only`, `undefined`). |
| 123–132 | `null_gaps`: sets `RSR = product(marginals)` and confirms both gaps are zero. This is the reproducibility check. |
| 135–155 | CLI for a single calculation. |

### 6.5 `exposure/mixture.py` (148 lines) — weighting conditions

| Lines | What happens |
|---|---|
| 17–25 | `_probability`: checks that a stored count and probability agree exactly. |
| 28–40 | `_log_lift` and `_lift_status`: define log lift; mark `zero_marginal`/`zero_joint` instead of returning `-inf`. |
| 43–127 | `summarize`: validate schema and weights; require subset-enabled report; one column per condition; check every subset entry is present and unique; recompute weighted joint exposure `sum_c w_c A[r,c]`, weighted marginals, pair joints, pair lifts, and composition gaps. |
| 120–127 | Output note: these are Monte Carlo plug-in estimates; uncertainty is **not** propagated. |
| 130–144 | CLI. |

### 6.6 `exposure/design.py` (174 lines) — the linear program

| Lines | What happens |
|---|---|
| 16–28 | `solve`: validate objective (`feasible` or `min-kl`), load rules, and run `summarize` on uniform weights as a validation device. |
| 29–62 | Validate the constraint specification: each entry needs `name`, `rule`, `kind` (`rule`/`predicate`/`pair`), `predicate_indices`, and at least one bound in [0,1]. |
| 63–78 | Build matrix rows: for each condition, look up the measured probability for the requested event. Lower bounds become `-row·w <= -lower`; upper bounds become `row·w <= upper`. |
| 80–82 | Solve LP: minimise 0 subject to `sum(w)=1`, `w >= min_weight`, and the bounds, using HiGHS. |
| 88–91 | Status 2 means infeasible; any other failure is a real error. |
| 92–127 | Optional minimum-KL objective to uniform, solved with SLSQP; handles exact equality rows that are redundant with the simplex. |
| 128–150 | Check finite weights and constraint violation ≤ 1e-7; report achieved exposures and slacks. |

This is the implementation of proposal §6: exposure is linear in mixture
weights, so design is a linear program.

### 6.7 `exposure/templates.py` (608 lines) — the 57-rule benchmark

| Lines | What happens |
|---|---|
| 21–51 | Constants: 3 objects, max 3 variables, `K_attr ∈ {1,2,3}`, `K_rel ∈ {0,1,2}`, expected cell sizes. Seven candidates in each `K_attr=1` cell so all four colours and three shapes appear; six elsewhere. |
| 53–89 | Hand-designed attribute bundles for 1, 2 and 3 attributes. |
| 91–108 | Single-relation, chain, and same-pair relation layouts. |
| 110–119 | Explicit limitations: provisional, not counterbalanced, no held-out split. |
| 146–160+ | `canonical_key`: alpha-renaming- and order-invariant semantic key, so `a red left_of b` and `x red left_of y` are the same rule. |
| rest | Builds the rule dict, checks grammar feasibility with `grammar_audit.assess`, and emits a manifest. |

The nine complexity cells are `attr1_rel0` … `attr3_rel2`. The benchmark is
*provisional*: `benchmark_audit` rejects it for confirmatory use.

### 6.8 `exposure/checker.py` (344 lines) — pixels back to objects

| Lines | What happens |
|---|---|
| 20–26 | `DetectionError` and `_PaletteError`: the checker abstains rather than guessing. |
| 28–87 | `_detect_palette`: require 64×64 RGB; compute distance of every pixel to the four palette colours; also check the pixel lies near a palette/white antialias blend; threshold; connected-component label; require exactly 3 components; require width/height in narrow ranges; classify shape by fill ratio; require ≥95% colour purity; reconstruct centres. |
| 90–100 | `_check_uncertain_centers`: for experimental narrow triangles, reject if two uncertain objects could swap spatial order. |
| 103–117 | Experimental narrow-triangle and attribute-only evaluators. |
| 120–154 | `_evaluate_uncertain`: three-valued logic; returns `None` (abstain) when centre intervals cannot settle a relation. |
| 157–179 | Experimental rule and variable-count evaluators (2–4 objects). |
| 182–189 | `_detect_sharpened`: accept mild sharpening only if two radii give identical labels and centres. |
| 192–233 | `detect`: try original palette pass; optionally correct a uniform grey background; try two thresholds and require agreement; optionally sharpen. This is a cascade of conservative fallbacks. |
| 236–244 | Fixed audit rules: four colours, three shapes, `red_left_of_square`, `blue_above_circle`. |
| 251–273 | `_detection_metrics`: conditional sensitivity/specificity (accepted images only), unconditional versions (detector failures count as misses), and coverage. |
| 276–317 | `audit`: render scenes, detect, compare pixel-derived satisfaction with annotation truth; count false accepts/rejects and detector failures. |
| 320–340 | CLI. |

The key limitation: it is calibrated to *clean renders*. On blurred generated
images it mostly **abstains** (attrition), which is why accepted-image error
rates look good but coverage is poor.

### 6.9 `exposure/benchmark_audit.py` (232 lines) — adversarial validity

| Lines | What happens |
|---|---|
| 29–31 | Relation kinds; binding classes; saturation threshold 0.9. |
| 34–43 | `_binding`: does the relation bind both endpoints, only subject, only object, or neither? |
| 46–52 | Analytic relation singleton exposure for the 3×3 renderer: `1 - (3 / C(9,3)) * (1/49) = 0.999271`. A relation is almost always true by accident. |
| 55–67 | Optional simulation to confirm the analytic value. |
| 70–128 | `audit`: group templates by `(K_attr, K_rel)`, count binding classes, detect cells where relation kind is perfectly confounded with binding. |
| 129–162 | Check all 12 colour–shape pairs for same-object co-binding; check held-out split metadata; emit blocking findings. |
| 163–201 | Always sets `confirmatory_ready: false` while blocking findings remain. |
| 204–228 | CLI; `--require-confirmatory` exits 2 when not ready. |

This module is the project's conscience. If you present the benchmark, present
this audit with it.

### 6.10 `exposure/pilot_train.py` (516 lines) — the diffusion model

| Lines | What happens |
|---|---|
| 47–67 | `_configure_determinism`: optional strict determinism; sets cuBLAS workspace, cuDNN flags, disables TF32; refuses if CUDA was already initialised without the setting. |
| 70–88 | `encode_rules`: each predicate becomes 4 integers: kind (colour/shape/left_of/above), subject slot, object slot, value. Slot 0 means padding. Max 3 object variables. |
| 91–104 | `Residual`: GroupNorm → SiLU → Conv → add time embedding → GroupNorm → SiLU → Conv → residual skip. |
| 107–155 | `PilotUNet`: learned time embedding; separate embeddings for kind/subject/object/value; a learned `null` token; three resolution levels (64→32→16); one multi-head cross-attention block at 16×16; skip connections on the way up. `drop` masks tokens per sample for classifier-free training; if a sample has no tokens, the null token is exposed so attention never sees an all-padding mask. |
| 158–162 | Linear beta schedule from 1e-4 to 0.02; `alpha_bar = cumprod(1-beta)`. |
| 229–371 | `train`: validate config; set determinism; load rules and weights; optionally restore v3 checkpoint (model, EMA, optimizer, RNG, runtime fingerprint); loop: sample condition by weight, render a fresh conditioned scene, run **all** rules on that scene to record incidental exposure, convert pixels to [-1,1], add noise at random timestep, drop condition with probability 0.1, predict noise, MSE loss, AdamW step, EMA update. Record counts, attempts, loss, timing, memory. |
| 352–371 | Checkpoint schema 3 contains config, rules, model, EMA, optimizer, report, and full RNG state plus code/runtime hashes. |
| 404–442 | `sample`: load EMA weights; start from Gaussian noise; loop t = T-1…0; predict noise; optional classifier-free guidance extrapolation; DDPM mean/variance update; clamp and convert to uint8 PNG. |
| 445–512 | CLI with `train`, `resume`, `sample` subcommands. |

Why online rendering matters: every training step sees a freshly rendered scene,
so "exposure" is a sampling probability, not dataset frequency, and memorisation
is reduced. The trade-off is that each step pays rendering cost.

### 6.11 Velocity / MOSAIC-inspired modules

| Module | Role |
|---|---|
| `velocity_probe.py` | Fixed-scene matched epsilon-vs-velocity experiment; velocity = `x1 - x0`, converted back to noise for sampling. |
| `velocity_online.py` | Online velocity training with schema-1 resumable checkpoints. |
| `velocity_guidance.py` | Classifier-free guidance in velocity space, then convert to noise. |
| `mosaic_pilotunet_probe.py` | Train `PilotUNet` on one MOSAIC-inspired cell (complexity × balance) using the existing token encoder and a velocity target. |
| `mosaic_pilotunet_warmstart.py` | Restore model/EMA/optimizer from an earlier checkpoint. Source checkpoints lacked RNG state, so this is a **warm start**, not exact resume; new checkpoints capture full RNG. |
| `mosaic_matched_cell.py` | Run the probe/warmstart for a selected cell in a fresh interpreter; records source hash. |
| `mosaic_distributed_supervisor.py` | Resume four-cell GPU training in bounded 10k segments across two machines. |
| `mosaic_continuous_supervisor.py` | Keep running bounded segments until a stop marker appears. |
| `mosaic_guidance_probe.py` | Paired guidance sweep from a frozen checkpoint. |
| `mosaic_toy_*` | Data feasibility, one-step denoising, sampler trajectory, conditioning checks on small toy models. |

The four MOSAIC-inspired cells are `{low, high} complexity × {balanced,
imbalanced}` concept distribution. This is explicitly **MOSAIC-inspired**, not a
replication: the renderer, architecture, objective and evaluator differ.

---

## 7. Every Python module, grouped

### 7.1 Core instrument (must know)

| File | Purpose |
|---|---|
| `rules.py` | Scene objects, attributes, relations, validation, satisfaction |
| `render.py` | Synthetic scene sampling, drawing, conditioned rejection sampling, CLI |
| `measure.py` | Monte Carlo condition × rule exposure matrix with Wilson intervals |
| `mixture.py` | Weighted exposure, log lift, pair lift |
| `composition.py` | Logit and log composition gaps, boundary handling |
| `design.py` | LP feasibility and minimum-KL mixture allocation |
| `templates.py` | 57-rule provisional benchmark |
| `grammar_audit.py` | Canonical rule layouts, topological feasibility, witnesses |
| `template_balance.py` | Per-cell vocabulary coverage report |
| `benchmark_audit.py` | Adversarial validity audit; `confirmatory_ready` |
| `validate_mixture.py` | Compare matrix predictions with fresh mixture samples |

### 7.2 Design and identifiability diagnostics

| File | Purpose |
|---|---|
| `design_subspace.py` | SVD of singleton-preserving equality directions |
| `design_sample.py` | Seeded hit-and-run sampling in the nullspace |
| `design_bounds.py` | Exact LP exposure/lift bounds and positive-lift corner |
| `design_chains.py` | Multi-chain split-R-hat and ESS diagnostics |
| `design_diagnostics.py` | Pooled correlations and VIF for rule-by-design rows |
| `design_strata.py` | Pooled vs within-rule vs within-stratum correlations |
| `design_identifiability.py` | Rank/identifiability of within-rule predictors |
| `design_holdout.py` | Evaluate fixed designs on a second measurement matrix |
| `design_extremes.py` | LP endpoint designs (min/max joint exposure) |
| `design_vif_floor.py` | Structural lower bound on VIF |
| `design_reference_explore.py` | Nonuniform reference-mixture sensitivity |
| `design_reference_*` | Reference diagnostics, endpoint diagnostics |
| `design_rule_allocation.py` | Nonuniform rule allocation exploration |
| `design_two_reference_exploration.py` | Two-reference designs to vary marginals within rule |
| `factorial_relations.py` | 24-rule factorial relational subset |
| `factorial_baseline.py` | Factorial baseline construction |

### 7.3 Generative models and training

| File | Purpose |
|---|---|
| `pilot_train.py` | `PilotUNet`, online DDPM training, sampling, resume |
| `velocity_online.py` | Online velocity/flow-matching trainer |
| `velocity_probe.py` | Fixed-scene epsilon vs velocity comparison |
| `velocity_guidance.py` | Velocity-space classifier-free guidance |
| `velocity_condition_probe.py` | Teacher-forced conditioning-use diagnostic |
| `mosaic_pilot_data.py` | Build MOSAIC-inspired toy datasets |
| `mosaic_pilotunet_probe.py` | One-cell PilotUNet velocity probe |
| `mosaic_pilotunet_warmstart.py` | Warm-start/resume with full RNG capture |
| `mosaic_matched_cell.py` | Select and run one matched cell |
| `mosaic_distributed_supervisor.py` | Multi-machine segmented training supervisor |
| `mosaic_continuous_supervisor.py` | Continuous segmented training supervisor |
| `mosaic_guidance_probe.py` | Paired guidance sampling |
| `mosaic_contrast.py` | Prepare/analyze the 2×2 matched contrast |
| `mosaic_multiscale_probe.py` | Larger multiscale learnability probe |
| `mosaic_resumable_probe.py` | Bounded resumable multiscale probe |
| `mosaic_toy_train.py` | Four-cell toy training |
| `mosaic_toy_denoise.py` | One-step denoising of held-out images |
| `mosaic_toy_sample.py` | Sample from toy models |
| `mosaic_toy_trajectory.py` | Reverse-trajectory diagnostic |
| `mosaic_toy_condition_probe.py` | Does changing the label change pixels? |
| `mosaic_toy_overfit.py` | Toy overfit check |
| `mosaic_preview_review_pack.py` | Blinded preview pack |
| `mosaic_fresh_*` / `mosaic_warmstart_probe.py` | Follow-up probes and reviews |
| `mosaic_step_probe.py` | Sampling-step comparison |

### 7.4 Diagnostics and probes

| File | Purpose |
|---|---|
| `denoise_probe.py` | One-step noise/reconstruction MSE vs zero baseline |
| `reconstruction_probe.py` | Truncated reverse reconstruction |
| `high_t_baseline.py` | One-step high-noise baseline (copy/zero/oracle) |
| `high_t_finetune.py` | Paired control vs high-t weighted fine-tuning |
| `identity_eps_finetune.py` | Standard vs identity-residual noise prediction |
| `time_embedding_probe.py` | Learned vs sinusoidal time embedding |
| `optimizer_probe.py` | AdamW vs momentum SGD on a fixed scene |
| `static_scene_probe.py` | Overfit one fixed scene |
| `gaussian_diagnostic.py` | 1D Gaussian-mixture Bayes denoiser reference |
| `gpu_preflight.py` | Read-only CUDA/GPU check |

### 7.5 Checker, audit, human review

| File | Purpose |
|---|---|
| `checker.py` | Pixel-based object/rule checker and clean-render audit |
| `checker_perturb.py` | Brightness/blur robustness audit |
| `checker_palette_probe.py` | Palette-tolerance diagnostic |
| `spatial_counterfactual_audit.py` | Clean negative/near-tie spatial cases |
| `human_audit.py` | Blinded review packs and scoring |
| `audit_agreement.py` | Pairwise reviewer agreement |
| `audit_followup.py` | Follow-up case resolution |
| `audit_block_comparison.py` | Paired checker coverage comparison |
| `audit_pilot_pack.py` | Build restricted pilot packs |
| `audit_pilot_score.py` | Score pilot reviews |
| `audit_validation_score.py` | Score validation reviews |
| `mosaic_guidance_predictions.py` | Freeze checker predictions for reviews |
| `mosaic_guidance_review_pack.py` | Build guidance review packs |
| `mosaic_guidance_review_score.py` | Score guidance reviews |

---

## 8. Every documentation file, grouped

The docs are unusually candid. For a presentation, cite the *question* each
document answers, not every number.

### 8.1 Orientation and specification

| Doc | What it answers |
|---|---|
| `README.md` | What this is, how to run it, doc index. |
| `docs/getting-started.md` | Install, test, first render. |
| `docs/architecture.md` | Module interfaces, design decisions, current limits. |
| `docs/experiment-spec.md` | Working semantics: events, bindings, exposure, gaps, pending decisions. **Read this before explaining any metric.** |
| `docs/research-and-hardware-plan.md` | Dataset strategy, CPU/CUDA plan, limitations. |
| `docs/review-2026-11-04.md` | Internal code/research review (with a correction notice). |
| `docs/verification-2026-11-04.md` | Independent re-derivation of the algebra and concerns. |
| `docs/stage0-status.md` | The master status file: every gate, every artifact, every caveat. |
| `docs/windows-gpu-access.md` | SSH, scheduled tasks, training tray. |

### 8.2 Rendering, rules, measurement

| Doc | What it answers |
|---|---|
| `docs/rendering-and-rules.md` | CLI options, rule JSON, predicate semantics, sampling, output JSON. |
| `docs/exposure-measurement.md` | Named rule sets, measurement CLI, satisfaction columns, Wilson intervals. |
| `docs/mixture-analysis.md` | Weighted exposures, rule/pair lift, interpretation limits. |
| `docs/composition-gap.md` | Both gap scales, boundary status, multiplicative null. |
| `docs/gaussian-diagnostic.md` | Analytic noise-loss reference for a 1D Gaussian mixture. |
| `docs/grammar-audit.md` | Canonical layouts and witnesses. |
| `docs/templates.md` | The provisional 57-template set. |
| `docs/template-balance.md` | Per-cell vocabulary coverage (not a balance claim). |
| `docs/benchmark-audit.md` | Relational validity findings and `confirmatory_ready`. |
| `docs/factorial-relations.md` | 24-rule factorial relation subset. |
| `docs/matched-marginal-pilot.md` | Low/high full-rule exposure with matched marginals. |

### 8.3 Exposure design and identifiability

| Doc | What it answers |
|---|---|
| `docs/exposure-design.md` | Feasibility constraints, LP formulation, solver statuses. |
| `docs/design-subspace.md` | Numerical singleton-preserving directions. |
| `docs/design-sample.md` | Hit-and-run sampling preserving singleton exposures. |
| `docs/design-bounds.md` | Exact LP exposure/lift bounds. |
| `docs/design-chains.md` | Multi-chain R-hat and ESS. |
| `docs/design-diagnostics.md` | Pooled correlations and VIF. |
| `docs/design-strata.md` | Correlation by rule and complexity. |
| `docs/design-identifiability.md` | Within-rule rank/identifiability. |
| `docs/design-holdout.md` | Fixed-design holdout. |
| `docs/design-extremes.md` | LP endpoint designs. |
| `docs/design-vif-floor.md` | Structural VIF lower bound. |
| `docs/design-reference-exploration.md` | Nonuniform reference-mixture sensitivity. |
| `docs/design-reference-design-diagnostics.md` | Sampled designs for that reference. |
| `docs/design-reference-endpoint-diagnostics.md` | Endpoint contributors to VIF. |
| `docs/design-reference-exploration.md` | Reference-mixture diagnostic intervention. |
| `docs/design-rule-allocation-exploration.md` | Nonuniform rule allocation. |
| `docs/design-two-reference-exploration.md` | Two-reference design to vary marginals within rule. |
| `docs/mixture-validation.md` | Matrix predictions vs fresh mixture samples. |

### 8.4 Checker and human audit

| Doc | What it answers |
|---|---|
| `docs/checker.md` | Prototype pixel checker and clean-render audit. |
| `docs/checker-perturb.md` | Coverage/error under brightness and blur. |
| `docs/checker-geometry-investigation.md` | Why valid-looking generated shapes get rejected. |
| `docs/checker-palette-diagnostic.md` | Whether relaxing the palette guard helps. |
| `docs/spatial-counterfactual-audit.md` | Clean negative/near-tie spatial cases. |
| `docs/human-audit.md` | Blinded pack preparation and scoring. |
| `docs/human-checklist.md` | Human decisions, access, image review, gate sign-off. |
| `docs/audit-pilot-pack.md` | The restricted four-condition 300-image pilot. |
| `docs/audit-block-comparison.md` | Paired checker coverage comparison. |
| `docs/pilot-audit-800000.md` | Human-reviewed pilot audit results. |
| `docs/calibration-310000.md` | Four-reviewer calibration at 310k updates. |

### 8.5 Training, GPU, MOSAIC

| Doc | What it answers |
|---|---|
| `docs/pilot-training.md` | The main training document: mechanics, probes, every measured result and caveat. |
| `docs/diffusion-failure-research.md` | Literature hypotheses for merged/blurry samples. |
| `docs/gpu-throughput-benchmark.md` | Short GPU speed screen before the overnight run. |
| `docs/mosaic-replication.md` | MOSAIC source review; no replication run. |
| `docs/mosaic-data-feasibility.md` | Can a matched 2×2 toy dataset be built? |
| `docs/mosaic-gpu-feasibility-81302.md` | Larger toy dataset + CUDA feasibility. |
| `docs/mosaic-pilotunet-feasibility-81373.md` | One-cell PilotUNet probe. |
| `docs/mosaic-pilotunet-warmstart.md` | Warm start vs exact resume. |
| `docs/mosaic-warmstart-probe.md` | 30k additional updates on one cell. |
| `docs/mosaic-resumable-probe.md` | Bounded resumable probe. |
| `docs/mosaic-multiscale-probe-81322.md` | Multiscale learnability probe. |
| `docs/mosaic-sampling-steps-81330.md` | 50 vs 200 vs 500 reverse steps. |
| `docs/mosaic-toy-conditioning.md` | Does the class condition change pixels? |
| `docs/mosaic-toy-denoise.md` | One-step denoising of known images. |
| `docs/mosaic-toy-trajectory.md` | Repeated reverse updates with an oracle. |
| `docs/mosaic-guidance-probe.md` | Classifier-free guidance feasibility. |
| `docs/mosaic-overnight-81390.md` | Overnight continuation to 150k. |
| `docs/mosaic-continuous-training.md` | Continuous segmented supervisor. |
| `docs/mosaic-760k-assessment.md` | Stopped 760k checkpoint assessment. |
| `docs/mosaic-fresh-760k-review.md` | Fresh 32-image review. |
| `docs/mosaic-matched-81570.md` | New bounded four-cell experiment. |
| `docs/mosaic-distributed-81570.md` | 500k-update distributed continuation. |

---

## 9. The existing 63-slide deck

`presentation/main.tex` / `main.pdf` is a 16:9 Beamer deck. It is technically
strong but **predates the latest MOSAIC training work** and is written as a
proposal-and-exploratory-work talk. Its four acts:

| Slides | Act | Content |
|---|---|---|
| 1–4 | Framing | Title, roadmap, adjacent work (MOSAIC, Okawa, GRASP, Bradley), contributions |
| 5–7 | Motivation | Combining ingredients vs complete arrangements; why it matters |
| 8–18 | Diffusion foundations | Pixels, noise, forward/reverse process, MSE objective, U-Net, cross-attention, conditioning, CFG, training vs generation |
| 19–30 | Scene world and exposure | Synthetic world, vocabulary, relations, predicates, distinct objects, exposure, hierarchy, counting, co-occurrence, lift, worked lift example |
| 31–39 | Evaluation and optimization | Checker success, independence reference, composition gap, zero gap, hypotheses, N50, learning curve, AdamW/EMA, DDPM vs DDIM |
| 40–47 | Design and audit | Fair comparison, exposure matrix, constrained optimization, controlled comparison, checker errors/abstentions, human audit, blinding |
| 48–55 | Evidence and obstacles | 800k pilot samples, work built, reviews, checker obstacle, relation saturation, design multicollinearity, Stage 0 gates |
| 56–63 | Implications and close | Next phase, positive/negative results, medical imaging, robotics, importance, take-home, questions, references |

Three numbers worth memorising for Q&A:

- ~54 templates in the proposal; the repo's provisional set has **57**.
- Stage 0 target checker error **≤ 2%**.
- Decorrelation target **|r| ≤ 0.3 and VIF < 2**.

Supporting files:

- `presentation/speaking-guide.md` — one bullet per slide (1–63).
- `presentation/detailed-speaker-notes.md` — technical narration and Q&A.
- `presentation/generate_noise.py` — deterministic noise-progression assets.
- `presentation/README.md` — build instructions (`make` or `tectonic main.tex`).

Build with:

```sh
cd presentation
make
```

If LaTeX is unavailable, present `main.pdf` directly and use this guide for the
code/docs parts the deck omits.

---

## 10. Stage 0: the dashboard you must show

| Gate | Proposal criterion | Current state | Why |
|---|---|---|---|
| Learnability | Reference RSR ~[0.3, 0.7]; overfit 1/10/100 scenes | **Not established** | Training reached 800k updates; previews mixed; no systematic learnability test |
| Checker | ≤2% error on generated images, per predicate/rule | **Not validated** | Clean renders fine; blur 1.0 rejects 84.5–90%; generated audits small/narrow |
| Decorrelation | `|r| ≤ 0.3`, `VIF < 2` | **Fails as written** | pooled `r = -0.446`, VIF `11.61`; `β_m` collinear with rule intercept |
| Design dose | Wide enough lift range to identify `β_λ` | **Narrow sample** | realized median range 0.0955 vs exact LP width 1.172; 23/57 rules admit positive lift but sampled chains reach none |
| Benchmark validity | relations not saturated, counterbalanced, held-out split | **Fails** | relation singleton 0.999271; binding confound; 3/12 same-object pairs; no split |
| MOSAIC replication | Adapted 2×2 contrast matches MOSAIC direction | **Not run** | Source review + MOSAIC-inspired pilot; no replication result |
| Human audit | ≥300 complexity-stratified generated images | **Incomplete** | 300-image restricted pilot covers 2 strata; four-condition, not full benchmark |

The proposal's go condition is: checker error ≤2%, design polytope meets
decorrelation, replication matches direction. None is met.

---

## 11. Caveats and likely audience questions

### "Is this novel?"
The proposal itself says novelty is not established and overlap with MOSAIC is
highest. The intended contributions are marginal-preserving exposure
manipulation, the composition gap, N50, transfer/interference measures, and
exposure allocation as constrained optimization. Say: "the contribution is the
controlled design, and it still needs full literature verification."

### "Why synthetic scenes?"
Because ground truth is exact and countable; natural-image labels are noisy and
confounded. The cost is external validity: geometric scenes may not transfer to
photographs.

### "Why is the model so small?"
It is an instrument, not a benchmark model. The proposal calls for a small U-Net
with capacity calibrated so median RSR lands in [0.3, 0.7]. Current width is 16
and images are 64×64.

### "Why does the checker reject so much?"
It hard-codes component count 3, widths 12, heights 11–12, fill-ratio ranges and
palette purity. Generated images violate those; the checker abstains. This is
attrition, not proven misclassification. The honest statement is: "we know
coverage is poor; conditional accuracy on accepted images is not the same as
validity."

### "If the relation singleton is 0.999271, can you study relations at all?"
Not with the current existential semantics and 3-object 3×3 world. The benchmark
audit blocks confirmatory use. The fix is to bind both relation endpoints to
attributed objects, counterbalance relation kind, cover all same-object pairs,
and target relation exposure in [0.2, 0.8].

### "What is the biggest risk?"
That the project passes instrument checks while no usable generator exists.
Stage 0 currently has no model criterion. A learnability pre-gate
(overfit 1 → 10 → 100 scenes → full mixture) is proposed.

### "What would a positive result look like?"
After controlling marginals, joint/rule exposure still predicts RSR; the
composition gap shrinks as lift rises; relational rules have larger slopes;
held-out rules need pairwise support.

### "What would a negative result look like?"
Joint exposure has negligible effect; models compose zero-shot from marginal
exposure; the independence null is adequate; allocation optimization has nothing
to exploit.

### "What has actually been measured?"
Clean-render checker audits, perturbation audits, estimated exposure matrices,
design diagnostics, many one-step training probes, small preview reviews, and a
restricted 300-image pilot audit. No confirmatory RSR/N50/gap estimate exists.

---

## 12. A 15-minute talk script

| Time | Slide / topic | Say |
|---|---|---|
| 0:00 | Title | "I'm presenting an instrument-development project asking which kind of training exposure makes image models follow combinations of instructions." |
| 0:45 | Question | "A model may know red and square separately yet fail at red left of square. Is the missing signal the ingredients, the pair, or the full rule?" |
| 2:00 | Why care | "Medical co-morbidity synthesis and safety-critical robotics need rare combinations, not just rare labels." |
| 3:00 | Synthetic world | "We use 64×64 scenes with three objects, four colours, three shapes, and two strict relations, so every rule can be counted exactly." |
| 4:30 | Exposure hierarchy | Define marginal, pair, full rule, lift. "Same marginals can hide radically different joint exposure." |
| 6:00 | Composition gap and N50 | "We compare full-rule success to the product of constituent successes; N50 asks how much exposure is needed to reach 50%." |
| 7:30 | Pipeline | Walk the nine boxes: render → measure → mixture → LP design → train → sample → checker → audit → gates. |
| 9:00 | What is built | Rule engine, renderer, measurement matrix, LP tools, PilotUNet, checker, human audit, MOSAIC-inspired 2×2. |
| 10:30 | What we found | "The instruments work, but Stage 0 has not passed: relation singletons saturate, design is collinear, checker coverage is poor, and the generator is not validated." |
| 12:30 | Honest status | Show the dashboard. "No confirmatory result exists; the negative/inconclusive evidence is the current contribution." |
| 14:00 | Next steps | Fix benchmark, pass learnability, validate checker, run MOSAIC replication, then Stage 1. |
| 15:00 | Close | "The central distinction is exposure to pieces versus exposure to combinations. The project has built the apparatus to measure that difference." |

If you have 30 minutes, add the diffusion math (slides 8–18), the design
optimization (slides 40–43), and the obstacles (slides 51–54).

---

## 13. Where to begin, hands-on

### Read in this order

1. `README.md` — 2 minutes.
2. `docs/getting-started.md` — 5 minutes.
3. `docs/experiment-spec.md` — 15 minutes; this is the semantic contract.
4. `exposure/rules.py` — 15 minutes.
5. `exposure/render.py` — 15 minutes.
6. `exposure/measure.py` — 20 minutes.
7. `exposure/mixture.py` + `exposure/composition.py` — 20 minutes.
8. `exposure/design.py` — 20 minutes.
9. `exposure/pilot_train.py` — 30 minutes (model + training + sampling).
10. `docs/stage0-status.md` — 20 minutes; read the caveats, not just the numbers.
11. `analysis_outputs/SUMMARY.md`, `RISKS.md`, `EVAL_CONTRACT.md` — prior analysis.
12. The proposal PDF, sections 1–6 and 11–12.

### Commands to verify the story live

```sh
# 1. Run the non-training test suite (training tests skip without torch)
uv run --no-sync python -m unittest discover -s tests -q

# 2. Render an unconditional scene
uv run python -m exposure.render --seed 42 --output outputs/demo-scene

# 3. Render a rule-conditioned scene
uv run python -m exposure.render --seed 42 \
  --rule examples/red_left_of_square.json --output outputs/demo-conditioned

# 4. Measure the exposure matrix (small, fast)
uv run python -m exposure.measure \
  --rules examples/exposure_rules.json --samples 200 \
  --include-subsets --output outputs/demo-exposure.json

# 5. Summarize a uniform mixture
uv run python -m exposure.mixture \
  --report outputs/demo-exposure.json \
  --weights examples/uniform_weights.json \
  --output outputs/demo-mixture.json

# 6. Check LP feasibility of the example constraints
uv run python -m exposure.design \
  --report outputs/demo-exposure.json \
  --constraints examples/exposure_constraints.json \
  --output outputs/demo-design.json

# 7. Audit the benchmark (expect confirmatory_ready: false)
uv run python -m exposure.benchmark_audit \
  --rules examples/exposure_rules.json \
  --output outputs/demo-benchmark-audit.json
```

Open `outputs/demo-conditioned.png` and `outputs/demo-conditioned.json` side by
side. That single pair explains the whole data model: pixels plus exact
annotations.

If torch is installed (`uv sync --extra train`), the training demo is:

```sh
uv run --extra train python -m exposure.pilot_train train \
  --rules examples/exposure_rules.json \
  --weights examples/uniform_weights.json \
  --output outputs/demo-pilot/pilot.pt \
  --steps 100 --batch-size 2 --seed 0 --device cpu

uv run --extra train python -m exposure.pilot_train sample \
  --checkpoint outputs/demo-pilot/pilot.pt \
  --rule red --output outputs/demo-pilot/red.png --seed 0 --device cpu
```

Do not present the 100-step sample as evidence of learning; it is a wiring test.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Rule** | Nonempty conjunction of attribute and relation predicates over distinct object variables. |
| **Predicate** | One attribute (`a is red`) or one relation (`a left_of b`). |
| **Binding** | Assignment of scene objects to rule variables; variables must map to distinct objects. |
| **Exposure** | Probability that a realized scene satisfies an event (rule or predicate set). |
| **Marginal** | Exposure of a single predicate. |
| **Joint exposure** | Exposure of a pair or full rule. |
| **Independence reference** | Product of constituent marginals. |
| **Lift** | `log(joint / product of marginals)`. |
| **Breadth** | Entropy-based diversity of partner contexts for a predicate. |
| **RSR** | Rule satisfaction rate of a model's generated images. |
| **Composition gap** | Difference between observed RSR and the independence reference, on logit or log scale. |
| **N50 / n50** | Exposure (count or probability) at which RSR reaches 50%. |
| **A-matrix** | Condition × event matrix of measured exposures. |
| **Simplex** | Set of nonnegative weights summing to one. |
| **LP** | Linear program; used for feasibility and bounds. |
| **Min-KL** | Closest feasible design to uniform in KL divergence. |
| **VIF** | Variance inflation factor; multicollinearity diagnostic. |
| **Stage 0** | Instrument-validation gate before confirmatory experiments. |
| **MOSAIC** | Prior work on scene complexity vs concept imbalance; the project plans a small adapted replication. |
| **CFG** | Classifier-free guidance; extrapolate between conditional and unconditional predictions. |
| **EMA** | Exponential moving average of model weights, used for sampling. |
| **DDPM** | Denoising diffusion probabilistic model; stochastic reverse process. |
| **DDIM / ODE** | Deterministic accelerated samplers. |
| **Attrition** | Checker rejects/abstains on many images rather than misclassifying them. |

---

## 15. One-slide summary

**Question:** Which training exposure level — predicate, pair, or full rule —
drives compositional success in conditional diffusion, after holding marginal
exposure fixed?

**Instrument:** a countable 64×64 three-object world; formal rule semantics; a
Monte Carlo exposure matrix; LP-based mixture design; a small conditional U-Net;
a pixel checker; blinded human audits.

**Evidence:** the CPU instruments work and are well tested. Training reached
800k updates and produced mixed previews. Every Stage 0 gate remains open:
learnability, checker validity, decorrelation, benchmark validity, MOSAIC
replication.

**Claim:** the project has built a controlled apparatus for a hard question, and
documents its own negative/inconclusive status rather than overclaiming.
