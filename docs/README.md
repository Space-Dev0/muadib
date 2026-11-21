# Documentation guide

This folder contains the design notes, measurements, audits, and status records
for the **Which Exposure Matters?** project. There are about 70 documents, so
they are grouped by **what you are trying to do**, not by date. You do not need
to read them all.

If you are overwhelmed, read the four files in
[§ The 15-minute path](#the-15-minute-path). Everything else is reference
material.

---

## What the project is, in three sentences

The project asks which level of training exposure — individual predicates,
pairs of predicates, or complete rules — is needed for a conditional diffusion
model to generate composed scenes correctly. It builds a controlled synthetic
world, formal rule semantics, an exposure-measurement matrix, linear-programming
design tools, a small diffusion model, and blinded auditing instruments. It is
still in **Stage 0 instrument development**, and its own status file says Stage 0
has **not passed**.

If you only remember one thing: **this is a carefully built apparatus plus
honest negative/inconclusive evidence, not a finished confirmatory study.**

---

## How the folders are organized

| Folder | Question it answers | Read it when |
|---|---|---|
| **`00-start-here/`** | What is this? How do I run it? What are the rules and limits? | First. |
| **`01-instruments/`** | How do we render scenes, define rules, measure exposure, and compute lift/gaps? | You need the core definitions. |
| **`02-benchmark/`** | What is the candidate 57-rule benchmark, and is it valid? | You need the rule set or its audit. |
| **`03-exposure-design/`** | How do we choose mixture weights so exposures are controlled and decorrelated? | You care about the experimental design. |
| **`04-checker-and-human-audit/`** | How do we turn generated pixels into pass/fail, and how do humans review them? | You care about evaluation validity. |
| **`05-training/`** | How does the pilot diffusion model train, sample, and fail? | You need the model/evidence story. |
| **`06-mosaic/`** | What MOSAIC-inspired experiments were run, and what do they show? | You need the closest prior work and replication status. |
| **`07-status-and-reviews/`** | What is the overall go/no-go state, and what was reviewed? | Before presenting or making claims. |
| **`08-operations/`** | How do we access the GPU laptop and benchmark throughput? | You are running the experiments. |

---

## The 15-minute path

Read these in order:

1. **[`00-start-here/getting-started.md`](00-start-here/getting-started.md)** —
   install, test, render one scene. (5 min)
2. **[`00-start-here/experiment-spec.md`](00-start-here/experiment-spec.md)** —
   the semantic contract: rules, distinct objects, exposure, gaps, pending
   decisions. (7 min)
3. **[`07-status-and-reviews/stage0-status.md`](07-status-and-reviews/stage0-status.md)** —
   what is actually established and what is not. Read the caveats, not just the
   numbers. (5 min)
4. **[`05-training/pilot-training.md`](05-training/pilot-training.md)** — skim
   the first 30 lines; know that training is exploratory and unvalidated. (3 min)

Then stop. You now know more than most readers of this repo.

---

## Reading paths by goal

### A. "I have to present this project"

1. [`00-start-here/getting-started.md`](00-start-here/getting-started.md)
2. [`00-start-here/experiment-spec.md`](00-start-here/experiment-spec.md)
3. [`01-instruments/rendering-and-rules.md`](01-instruments/rendering-and-rules.md)
4. [`01-instruments/exposure-measurement.md`](01-instruments/exposure-measurement.md)
5. [`01-instruments/composition-gap.md`](01-instruments/composition-gap.md)
6. [`07-status-and-reviews/stage0-status.md`](07-status-and-reviews/stage0-status.md)
7. [`05-training/pilot-training.md`](05-training/pilot-training.md) (first third)
8. [`04-checker-and-human-audit/checker.md`](04-checker-and-human-audit/checker.md)
9. [`03-exposure-design/design-diagnostics.md`](03-exposure-design/design-diagnostics.md)
10. [`02-benchmark/benchmark-audit.md`](02-benchmark/benchmark-audit.md)
11. [`06-mosaic/mosaic-replication.md`](06-mosaic/mosaic-replication.md)

Also read `presentation/PROJECT-WALKTHROUGH.md` and use
`presentation/main.pdf` for slides.

**The four numbers to know:**

- relation singleton exposure **0.999271** (relations are almost always true by
  accident),
- decorrelation **r = -0.446, VIF = 11.61** against targets **|r| ≤ 0.3, VIF < 2**,
- checker rejection **84.5–90%** at blur radius 1.0,
- **Stage 0 not passed; no confirmatory result.**

### B. "I need to use the code"

1. [`00-start-here/getting-started.md`](00-start-here/getting-started.md)
2. [`01-instruments/rendering-and-rules.md`](01-instruments/rendering-and-rules.md)
3. [`01-instruments/exposure-measurement.md`](01-instruments/exposure-measurement.md)
4. [`01-instruments/mixture-analysis.md`](01-instruments/mixture-analysis.md)
5. [`03-exposure-design/exposure-design.md`](03-exposure-design/exposure-design.md)
6. [`05-training/pilot-training.md`](05-training/pilot-training.md)
7. [`08-operations/windows-gpu-access.md`](08-operations/windows-gpu-access.md)
   only if you are on the GPU laptop.

Then read the actual code in this order:
`exposure/rules.py` → `render.py` → `measure.py` → `mixture.py` →
`composition.py` → `design.py` → `pilot_train.py` → `checker.py`.

### C. "I need to judge the research rigor"

1. [`00-start-here/experiment-spec.md`](00-start-here/experiment-spec.md)
2. [`07-status-and-reviews/stage0-status.md`](07-status-and-reviews/stage0-status.md)
3. [`07-status-and-reviews/review-2026-11-04.md`](07-status-and-reviews/review-2026-11-04.md)
4. [`07-status-and-reviews/verification-2026-11-04.md`](07-status-and-reviews/verification-2026-11-04.md)
5. [`02-benchmark/benchmark-audit.md`](02-benchmark/benchmark-audit.md)
6. [`04-checker-and-human-audit/checker-perturb.md`](04-checker-and-human-audit/checker-perturb.md)
7. [`03-exposure-design/design-vif-floor.md`](03-exposure-design/design-vif-floor.md)
8. [`03-exposure-design/design-identifiability.md`](03-exposure-design/design-identifiability.md)
9. [`06-mosaic/mosaic-replication.md`](06-mosaic/mosaic-replication.md)

The important distinction: **instrument measurements are not research
findings.** The docs are careful about this; preserve that distinction.

### D. "I only care about MOSAIC / prior work"

1. `/adjacent-research/README.md` — MOSAIC paper notes and citation.
2. [`06-mosaic/mosaic-replication.md`](06-mosaic/mosaic-replication.md) — source
   review and outstanding prerequisites.
3. [`06-mosaic/mosaic-data-feasibility.md`](06-mosaic/mosaic-data-feasibility.md)
4. [`06-mosaic/mosaic-matched-81570.md`](06-mosaic/mosaic-matched-81570.md)
5. [`06-mosaic/mosaic-distributed-81570.md`](06-mosaic/mosaic-distributed-81570.md)
6. [`07-status-and-reviews/stage0-status.md`](07-status-and-reviews/stage0-status.md)
   — the MOSAIC gate row.

**Status:** MOSAIC has been reviewed and imitated, not replicated. Do not claim
a replication.

---

## Recommended order for a first full pass

If you want the whole story without reading every document:

```
00-start-here/getting-started.md
        ↓
00-start-here/experiment-spec.md        ← semantics
        ↓
01-instruments/rendering-and-rules.md
01-instruments/exposure-measurement.md
01-instruments/mixture-analysis.md
01-instruments/composition-gap.md
        ↓
02-benchmark/templates.md
02-benchmark/benchmark-audit.md          ← is the benchmark usable? (no)
        ↓
03-exposure-design/exposure-design.md
03-exposure-design/design-diagnostics.md ← can we decorrelate? (not yet)
        ↓
05-training/pilot-training.md            ← does the model work? (not yet)
        ↓
04-checker-and-human-audit/checker.md    ← can we measure success? (not yet)
        ↓
06-mosaic/mosaic-replication.md          ← closest prior work
        ↓
07-status-and-reviews/stage0-status.md   ← the verdict
```

After that, read the numbered design docs in `03-exposure-design/` only when you
need a specific diagnostic.

---

## How to read the documents: conventions

### Status words

The docs use the same words consistently. Treat them as a legend:

| Word | Means |
|---|---|
| **Current / implemented** | Code exists and is tested. Does **not** mean research-validated. |
| **Exploratory / diagnostic** | A small, non-confirmatory check. Useful for debugging, not for claims. |
| **Pilot** | An early training or audit run. Not a result. |
| **Not passed** | A Stage 0 gate is unmet. Do not proceed to confirmatory claims. |
| **Provisional** | A candidate artifact (e.g. the 57 templates) that may change. |
| **Historical** | Superseded but retained for the record (e.g. the review correction). |

### The honesty rules

1. **No document is a preregistration.** The experiment specification explicitly
   says so.
2. **Monte Carlo estimates are plug-in estimates.** Wilson intervals are
   pointwise and are not propagated into mixture or design calculations.
3. **Zero empirical counts are not proof of impossibility.** The docs mark
   `zero_joint` / `zero_marginal` instead of treating zeros as exact.
4. **Design feasibility applies to the estimated matrix, not the population.**
5. **A checker abstention is not a model error.** Accepted-image error rates and
   coverage are reported separately.
6. **No human labels are fabricated.** If human judgments are missing, the docs
   say so.
7. **No Stage 0 gate is passed.** If a document seems to imply otherwise, check
   `07-status-and-reviews/stage0-status.md`.

### If a number looks surprising

Most numbers in these docs are from one of four sources:

- **analytic** (e.g. relation singleton 0.999271),
- **Monte Carlo on the prototype renderer** (e.g. exposure matrices),
- **a training diagnostic** (e.g. one-step noise MSE),
- **a human review pack** (e.g. 300-image pilot).

They are not interchangeable. The doc usually names the source. If it does not,
treat the number as exploratory.

---

## Full index

### `00-start-here/` — orientation

| Doc | One-line purpose |
|---|---|
| [`getting-started.md`](00-start-here/getting-started.md) | Install, run tests, render a first scene. |
| [`architecture.md`](00-start-here/architecture.md) | Module interfaces, design decisions, current limits. |
| [`experiment-spec.md`](00-start-here/experiment-spec.md) | Working semantics for rules, exposure, gaps, and pending decisions. **Start here for definitions.** |
| [`research-and-hardware-plan.md`](00-start-here/research-and-hardware-plan.md) | Dataset strategy, CPU/CUDA plan, limitations. |

### `01-instruments/` — core measurement

| Doc | One-line purpose |
|---|---|
| [`rendering-and-rules.md`](01-instruments/rendering-and-rules.md) | CLI, rule JSON, predicate semantics, sampling, output JSON. |
| [`exposure-measurement.md`](01-instruments/exposure-measurement.md) | Measurement CLI, condition × rule matrix, Wilson intervals. |
| [`mixture-analysis.md`](01-instruments/mixture-analysis.md) | Weighted exposures, rule/pair lift, interpretation limits. |
| [`composition-gap.md`](01-instruments/composition-gap.md) | Both gap scales, boundary status, multiplicative null. |
| [`gaussian-diagnostic.md`](01-instruments/gaussian-diagnostic.md) | Analytic noise-loss reference for a 1D Gaussian mixture. |
| [`grammar-audit.md`](01-instruments/grammar-audit.md) | Canonical rule layouts and feasibility witnesses. |

### `02-benchmark/` — rule set and validity

| Doc | One-line purpose |
|---|---|
| [`templates.md`](02-benchmark/templates.md) | The provisional 57-template benchmark across 9 cells. |
| [`template-balance.md`](02-benchmark/template-balance.md) | Per-cell vocabulary coverage; not a balance claim. |
| [`benchmark-audit.md`](02-benchmark/benchmark-audit.md) | Adversarial validity audit; `confirmatory_ready: false`. |
| [`factorial-relations.md`](02-benchmark/factorial-relations.md) | 24-rule factorial relation subset. |
| [`matched-marginal-pilot.md`](02-benchmark/matched-marginal-pilot.md) | Low/high full-rule exposure with two matched marginals. |

### `03-exposure-design/` — LP, sampling, diagnostics

| Doc | One-line purpose |
|---|---|
| [`exposure-design.md`](03-exposure-design/exposure-design.md) | Feasibility constraints, LP formulation, solver statuses. |
| [`design-subspace.md`](03-exposure-design/design-subspace.md) | SVD of singleton-preserving equality directions. |
| [`design-sample.md`](03-exposure-design/design-sample.md) | Hit-and-run sampling inside the nullspace. |
| [`design-bounds.md`](03-exposure-design/design-bounds.md) | Exact LP exposure/lift bounds and positive-lift corner. |
| [`design-chains.md`](03-exposure-design/design-chains.md) | Multi-chain R-hat and ESS diagnostics. |
| [`design-diagnostics.md`](03-exposure-design/design-diagnostics.md) | Pooled correlations and VIF. |
| [`design-strata.md`](03-exposure-design/design-strata.md) | Pooled vs within-rule vs within-stratum correlations. |
| [`design-identifiability.md`](03-exposure-design/design-identifiability.md) | Within-rule rank/identifiability of predictors. |
| [`design-holdout.md`](03-exposure-design/design-holdout.md) | Fixed designs evaluated on a second matrix. |
| [`design-extremes.md`](03-exposure-design/design-extremes.md) | LP endpoint designs. |
| [`design-vif-floor.md`](03-exposure-design/design-vif-floor.md) | Structural lower bound on VIF. |
| [`design-reference-exploration.md`](03-exposure-design/design-reference-exploration.md) | Nonuniform reference-mixture sensitivity. |
| [`design-reference-design-diagnostics.md`](03-exposure-design/design-reference-design-diagnostics.md) | Sampled designs for that reference. |
| [`design-reference-endpoint-diagnostics.md`](03-exposure-design/design-reference-endpoint-diagnostics.md) | Endpoint contributors to VIF. |
| [`design-rule-allocation-exploration.md`](03-exposure-design/design-rule-allocation-exploration.md) | Nonuniform rule allocation. |
| [`design-two-reference-exploration.md`](03-exposure-design/design-two-reference-exploration.md) | Two-reference design to vary marginals within rule. |
| [`mixture-validation.md`](03-exposure-design/mixture-validation.md) | Matrix predictions vs fresh mixture samples. |

### `04-checker-and-human-audit/` — evaluation validity

| Doc | One-line purpose |
|---|---|
| [`checker.md`](04-checker-and-human-audit/checker.md) | Prototype pixel checker and clean-render audit. |
| [`checker-perturb.md`](04-checker-and-human-audit/checker-perturb.md) | Coverage/error under brightness and blur. |
| [`checker-geometry-investigation.md`](04-checker-and-human-audit/checker-geometry-investigation.md) | Why valid-looking generated shapes get rejected. |
| [`checker-palette-diagnostic.md`](04-checker-and-human-audit/checker-palette-diagnostic.md) | Whether relaxing the palette guard helps. |
| [`spatial-counterfactual-audit.md`](04-checker-and-human-audit/spatial-counterfactual-audit.md) | Clean negative/near-tie spatial cases. |
| [`human-audit.md`](04-checker-and-human-audit/human-audit.md) | Blinded review pack preparation and scoring. |
| [`human-checklist.md`](04-checker-and-human-audit/human-checklist.md) | Human decisions and Stage 0 sign-off. |
| [`audit-pilot-pack.md`](04-checker-and-human-audit/audit-pilot-pack.md) | Build the restricted four-condition 300-image pilot. |
| [`audit-block-comparison.md`](04-checker-and-human-audit/audit-block-comparison.md) | Paired checker coverage comparison. |
| [`pilot-audit-800000.md`](04-checker-and-human-audit/pilot-audit-800000.md) | Human-reviewed pilot audit results. |
| [`calibration-310000.md`](04-checker-and-human-audit/calibration-310000.md) | Four-reviewer calibration at 310k updates. |

### `05-training/` — model and evidence

| Doc | One-line purpose |
|---|---|
| [`pilot-training.md`](05-training/pilot-training.md) | Main training document: mechanics, probes, results, caveats. |
| [`diffusion-failure-research.md`](05-training/diffusion-failure-research.md) | Literature hypotheses for merged/blurry samples. |

### `06-mosaic/` — MOSAIC-inspired work

| Doc | One-line purpose |
|---|---|
| [`mosaic-replication.md`](06-mosaic/mosaic-replication.md) | MOSAIC source review; no replication run. |
| [`mosaic-data-feasibility.md`](06-mosaic/mosaic-data-feasibility.md) | Can a matched 2×2 toy dataset be built? |
| [`mosaic-gpu-feasibility-81302.md`](06-mosaic/mosaic-gpu-feasibility-81302.md) | Larger toy dataset + CUDA feasibility. |
| [`mosaic-pilotunet-feasibility-81373.md`](06-mosaic/mosaic-pilotunet-feasibility-81373.md) | One-cell PilotUNet probe. |
| [`mosaic-pilotunet-warmstart.md`](06-mosaic/mosaic-pilotunet-warmstart.md) | Warm start vs exact resume. |
| [`mosaic-warmstart-probe.md`](06-mosaic/mosaic-warmstart-probe.md) | 30k additional updates on one cell. |
| [`mosaic-resumable-probe.md`](06-mosaic/mosaic-resumable-probe.md) | Bounded resumable probe. |
| [`mosaic-multiscale-probe-81322.md`](06-mosaic/mosaic-multiscale-probe-81322.md) | Multiscale learnability probe. |
| [`mosaic-sampling-steps-81330.md`](06-mosaic/mosaic-sampling-steps-81330.md) | 50 vs 200 vs 500 reverse steps. |
| [`mosaic-toy-conditioning.md`](06-mosaic/mosaic-toy-conditioning.md) | Does the class condition change pixels? |
| [`mosaic-toy-denoise.md`](06-mosaic/mosaic-toy-denoise.md) | One-step denoising of known images. |
| [`mosaic-toy-trajectory.md`](06-mosaic/mosaic-toy-trajectory.md) | Repeated reverse updates with an oracle. |
| [`mosaic-guidance-probe.md`](06-mosaic/mosaic-guidance-probe.md) | Classifier-free guidance feasibility. |
| [`mosaic-overnight-81390.md`](06-mosaic/mosaic-overnight-81390.md) | Overnight continuation to 150k. |
| [`mosaic-continuous-training.md`](06-mosaic/mosaic-continuous-training.md) | Continuous segmented supervisor. |
| [`mosaic-760k-assessment.md`](06-mosaic/mosaic-760k-assessment.md) | Stopped 760k checkpoint assessment. |
| [`mosaic-fresh-760k-review.md`](06-mosaic/mosaic-fresh-760k-review.md) | Fresh 32-image review. |
| [`mosaic-matched-81570.md`](06-mosaic/mosaic-matched-81570.md) | New bounded four-cell experiment. |
| [`mosaic-distributed-81570.md`](06-mosaic/mosaic-distributed-81570.md) | 500k-update distributed continuation. |

### `07-status-and-reviews/` — the verdict

| Doc | One-line purpose |
|---|---|
| [`stage0-status.md`](07-status-and-reviews/stage0-status.md) | Master status: every gate, artifact, and caveat. **Read before presenting.** |
| [`review-2026-11-04.md`](07-status-and-reviews/review-2026-11-04.md) | Internal code/research review, with a correction notice. |
| [`verification-2026-11-04.md`](07-status-and-reviews/verification-2026-11-04.md) | Independent re-derivation of the algebra and concerns. |

### `08-operations/` — running it

| Doc | One-line purpose |
|---|---|
| [`windows-gpu-access.md`](08-operations/windows-gpu-access.md) | SSH, scheduled tasks, training tray. |
| [`gpu-throughput-benchmark.md`](08-operations/gpu-throughput-benchmark.md) | Short GPU speed screen before the overnight run. |

---

## Where things live outside `docs/`

| Path | What it contains |
|---|---|
| `README.md` | Project entry point and quick start. |
| `presentation/PROJECT-WALKTHROUGH.md` | Complete presenter's guide: basics, pipeline, code, docs. |
| `presentation/main.pdf` | 66-page Beamer deck. |
| `presentation/speaking-guide.md` | One bullet per slide. |
| `presentation/detailed-speaker-notes.md` | Technical narration and Q&A. |
| `adjacent-research/README.md` | MOSAIC paper notes and citation. |
| `analysis_outputs/` | Prior static analysis: summary, risks, evaluation contract. |
| `examples/` | Small rule/weight/constraint JSON files. |
| `exposure/` | All Python modules. |
| `tests/` | Unit and CLI tests. |
| `scripts/` | Windows/GPU launchers and supervisors. |

---

## Maintaining this folder

When adding a document:

1. Put it in the folder matching its purpose, not in the `docs/` root.
2. Use a descriptive `kebab-case` filename.
3. Add it to the full index above.
4. If it introduces a new category, add the folder to the table in
   [§ How the folders are organized](#how-the-folders-are-organized).
5. Mark exploratory or historical work explicitly in the first paragraph.

The goal is that a new reader can answer three questions without opening more
than three files:

- **What is the question?** → `00-start-here/experiment-spec.md`
- **What is actually established?** → `07-status-and-reviews/stage0-status.md`
- **Where is the code for it?** → `presentation/PROJECT-WALKTHROUGH.md`, §6.
