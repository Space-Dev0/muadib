# Stage 0 status

The current rule and renderer source changes alter the source fingerprints
recorded in earlier online pilot checkpoints. Exact resume of those checkpoints
requires their original code revision and compatible runtime; do not bypass the
fingerprint check. Sampling compatibility should be verified separately.

**Not passed.** The following measurements are instrument-development results,
not confirmatory research findings.

A four-cell CUDA multiscale feasibility run has produced 16 exploratory
previews and a blinded team review pack; both checkers abstained on all 16
target-circle questions. It is one training-seed block, not a MOSAIC gate
result. See [GPU probe](mosaic-multiscale-probe-81322.md).

The fresh 40-image restricted checker validation has been reviewed: the
standard checker answered 19/40 and the experimental candidate 28/40, with
zero disagreements among answered questions. Twelve candidate decisions
abstained. This small, four-rule sample does not establish the checker gate;
see [checker validation](checker-geometry-investigation.md#fresh-validation-results).

The restricted 300-image, four-condition pilot has now been human-reviewed.
The standard checker classified 95/300 and abstained on 205; the sample covers
only two complexity strata and does not establish the full checker gate. See
[pilot audit](pilot-audit-800000.md) for agreement, per-rule coverage, errors,
and follow-up cases. The six submitted follow-up answers resolve the selected
cases for a separate sensitivity summary, but do not change original reviews,
reduce abstention, or establish the gate.

For the current fixed-singleton, equally weighted complete rule rows, the
correlation between log mean marginal exposure and relation count implies a
pairwise VIF lower bound of 3.478 if both are included with an intercept. This
is a property of the estimated pilot matrix and specified analysis, not a
population or gate result; see [VIF floor](design-vif-floor.md). A fitted nonuniform reference reduces
the exploratory log-marginal/lift correlation, but five sampled chains
still give held-out lift VIF 5.997–6.233, above 2; see
[reference design diagnostics](design-reference-design-diagnostics.md).
A further [rule-allocation exploration](design-rule-allocation-exploration.md)
achieves held-out VIF below 2 for four features, but changes the rule-weighted
population, uses concentrated endpoint mixtures, omits breadth, and leaves the
within-rule marginal effect unidentified. It is not a gate assessment.
A separate
[one-step MOSAIC toy diagnostic](mosaic-toy-denoise.md) evaluates known images,
not generated-image success or replication.

## Latest checkpoint and calibration (2026-11-08)

The online velocity pilot reached 100,000 updates on the GPU laptop. A local
16-image preview covers four conditions and two complexity strata; all 16
images were rejected by the current checker. This is a small convenience preview,
not a model-quality estimate or the proposal's 300-image checker audit. A
reviewer-only calibration pack is available in the ignored local path
`outputs/preview-review-pack-100000.zip`; it contains no checker predictions.
Each reviewer can independently answer the 16 rule questions and 32 separate
predicate questions. Actual human judgments remain outstanding. The laptop was
unreachable over SSH when this pack was prepared, so no newer checkpoint was
verified. See [human checklist](human-checklist.md) and
[pilot training](pilot-training.md#evidence-and-omissions).

On 2026-11-09 the GPU laptop was reachable again and contained the completed
110,000-update checkpoint. A sign-in tray task continued training through
120,000 updates and began the segment toward 130,000. A matched preview of 16 images (four conditions,
seeds 101–104) was copied locally to `outputs/preview-110000-20261109/`.
The prototype checker rejected all 16: eight for palette deviations, three
for ambiguous shapes, four for two detected components and one for unsupported
size or occlusion. These are detector rejections, not measured human error
rates or evidence that training has improved or worsened the model. The human
audit, decorrelation gate, and MOSAIC replication remain outstanding.

On 2026-11-10 the laptop had a completed 310,000-update checkpoint and was
training toward 320,000. A matched preview used the same four conditions and
seeds 101–104. The checker accepted 5/16 images and rejected 11: seven for
unsupported size or occlusion, three for four detected components, and one
for an ambiguous shape. Its 110,000-update counterpart accepted 0/16. This
small comparison measures checker coverage, not conditional accuracy, and
does not establish a population-level improvement. Images and predictions
are in ignored `outputs/preview-310000-20261110/`; the reviewer-only pack is
`outputs/preview-review-pack-310000.zip`. No human judgments were supplied or
inferred. Four rules and two complexity strata cannot substitute for the
formal audit.

On 2026-11-11 four reviewers returned complete 310,000-preview calibration
files. They agreed unanimously on 14/16 whole-rule and 31/32 predicate
questions. The checker rejected 11/16 whole-rule images, and one of the five
accepted images originally had a 2–2 human disagreement. The team subsequently
adjudicated all three disputed questions as yes; this does not increase the
sample size or establish checker accuracy. See
[calibration results](calibration-310000.md); this does not pass the checker
gate. The GPU was actively training from the 760,000-update checkpoint toward
770,000 when checked.
Four exploratory samples from 760,000 updates (one per pilot condition, seed
101) contain recognizable shapes; the blue-above-circle sample has no clear
circle and the red sample shows four objects. The checker accepted one image
and rejected the other three. This is a four-image diagnostic, not an
accuracy estimate or comparison of training checkpoints. Local copies are in
ignored `outputs/preview-760000-20261111/`.

At the user's request training was paused with a persistent pause marker on
2026-11-11. The latest saved checkpoint was 800,000 updates. Four seed-101
samples, one per pilot condition, were generated from it; the checker accepted
one and rejected three for unrecognized foreground colors. These four images
are an exploratory diagnostic, not a quality estimate or Stage 0 evaluation.
Local copies are in ignored `outputs/preview-800000-20261111/`. The checker,
design decorrelation, and MOSAIC gates remain outstanding.

## Blocking status (verified 2026-11-05)

Stage 0 currently has **no learnability criterion**, so it could be "passed" by
instrument checks while no usable generator exists. The verified state is:

| Gate | Current state | Evidence |
| --- | --- | --- |
| Learnability | **Not established** | The four-rule velocity pilot reached 800,000 updates and generated recognizable shapes. Small human-reviewed previews exist, but no systematic learnability test, full benchmark model, or overfit-1/10/100-scene gate has been completed. |
| Relational validity | **Fails** | Analytic relation singleton exposure `0.999271`; `left_of` is subject-only `3/3` and `above` both-bound `3/3` in `K_attr=2, K_rel=1`. `benchmark_audit` sets `confirmatory_ready: false`. |
| Decorrelation | **Fails as written** | Pooled `r = −0.446`, VIF `11.61` vs proposal `|r| ≤ 0.3`, `VIF < 2`; `β_m` is exactly collinear with the rule intercept. |
| Design dose | **Narrow sample, wider polytope** | Realized median within-rule range `0.0955`; exact LP width `1.172`; `23/57` rules admit positive lift but no sampled chain reaches the corner. |
| Checker on generated images | **Not validated** | Human-reviewed generated-image calibrations are available; substantial rejection remains, and their samples are too small and narrow to establish the 2% criterion. The experimental geometry detector is not validated for research use. |
| MOSAIC replication | **Not run** | Analysis plumbing and a small data-feasibility pilot exist; no adaptation models have been trained or evaluated. |

A learnability pre-gate should require: overfit one fixed scene (visual match
and low MSE), overfit 10 scenes (RSR ≥ 0.9), overfit 100 scenes (RSR ≥ 0.7),
then the full template mixture (median reference RSR ≥ 0.3 with a
capacity/budget sweep). If it fails, the failure is the result; do not proceed
to exposure experiments.

A matched fixed-scene velocity-prediction experiment produced visible colored
shapes, unlike the matched epsilon samples, but did not reliably reproduce the
three-object target. This small diagnostic does not validate generated images
or pass the stage; see [pilot training](pilot-training.md#matched-velocity-prediction-fixed-scene-experiment).
An additional 4,000-update online velocity pilot on varied scenes produced two
exploratory samples with colored blobs but no clearly identifiable shapes.
A fresh checkpointed online velocity run reached 8,000 updates; its two
inspected samples also showed colored blobs without clear shapes or reliable
relations, so the human audit remains blocked on suitable generated images.
At 12,000 updates the same two sampled conditions appeared cleaner but still
did not clearly satisfy their requested conditions.
A small matched guidance sweep at scales 1, 2 and 4 altered color and sometimes
suppressed objects; the 12 inspected images remained blurry and did not
consistently show the requested condition.

A fixed-design holdout using a second 100-scene-per-condition measurement
matrix gave pooled correlation -0.4640 and log-lift VIF 11.07; the maximum
singleton deviation from the uniform reference was 0.01767. This exploratory
analysis omits breadth and does not assess the final gate. See
[design holdout](design-holdout.md).

| Component | Current evidence | Remaining work |
| --- | --- | --- |
| Renderer and rules | 2D renderer, annotation-based rule evaluation and grammar witnesses | Finalize and validate the benchmark vocabulary and full counterbalancing. |
| Candidate templates | Provisional 57-template generator covers all singleton colors and shapes in each one-attribute cell; a 100-scene-per-template sampling smoke test completed, but only `3/12` color–shape pairs are co-bound and relation kind is confounded with endpoint binding | Redesign as a factorial, counterbalanced set with a held-out split; `benchmark_audit` currently reports `confirmatory_ready: false`. |
| Pilot training | Locked sync and 127 Windows tests passed. A deterministic pilot reached 12,000 steps with checkpoints, and a 4,000-step fixed-scene diagnostic ran on RTX 3050 with truncated-reconstruction, one-step high-t baseline, and paired high-t fine-tuning follow-ups. | Inspected online samples were noise; single fixed-scene sample was mostly dark noise (MSE 0.8017); no validated model outputs, reference model, controlled guidance sweep, or human audit. See [pilot training](pilot-training.md). |
| Checker | Clean-render and controlled perturbation audits; conservative sharpening fallback; per-rule rejection, coverage and conditional/unconditional sensitivity/specificity now reported | Rejection is `84.5–90%` at blur radius `1.0`; accepted images show zero observed rule errors, so the failure is attrition. Validate on generated images and audit with human judgments. |
| Exposure matrix and design | Monte Carlo estimates, LP feasibility and numerical minimum-KL allocation | Quantify matrix uncertainty and verify feasible exposure manipulations and leave-out probes on the final benchmark. |
| Decorrelation | Twenty exploratory hit-and-run designs from the 57-condition pilot; pooled correlations and VIF computed over 1,140 rule-by-design rows; exact LP lift bounds and multi-chain diagnostics now available | Pooled `r = −0.446`, VIF `11.61` fails the proposal gate as written; `β_m` is exactly collinear with the rule intercept. Define the gate on an identified within-rule quantity and target `β_λ`/`β_λR`. |
| Gaussian diagnostic | One-dimensional analytic Bayes noise-loss reference and Monte Carlo check | Connect normalization checks to the eventual model/loss implementation. |
| MOSAIC contrast | Preliminary paper/code review | Run and document the adapted replication; see [MOSAIC notes](mosaic-replication.md). |
| Human audit | Blinded preparation, prediction and scoring for rules and independently judged predicate events | Obtain model-generated images and actual independent human judgments; none have been fabricated. |

For the current checker, a 200-scene holdout on separate synthetic seeds
41000–41199 yielded coverage 200/200 for clean and 0.90 brightness, 114/200 for
brightness 1.10 and blur radius 0.5, and 31/200 for blur radius 1.0. The
previous implementation baseline on these seeds was 85/200 for blur radius 0.5.
No rule errors were observed on accepted examples;
rejected scenes are excluded, so these counts do not establish overall checker
validity. These figures replace the earlier 100-scene results from an earlier
checker version. Reproduce with:

```sh
uv run python -m exposure.checker_perturb --samples 200 --seed 41000 --output outputs/checker-perturb-41000.json
```

The checker's sharpening fallback is attempted only when the original palette
validation passes. It accepts
only if two mild `UnsharpMask` settings with radii `0.5` and `0.6` and percent
`100` return identical object labels and exact centers. The fallback does not
validate generated images, pass Stage 0, or ensure robustness.

The proposal's checker, design-decorrelation and MOSAIC gates must be evaluated
using the agreed final instruments and data. No Stage 1 pilot result exists.
CPU pilot tests and the mechanical
sample have passed, and the laptop's isolated environment has `uv` 0.13.6,
PyTorch 2.9.1+cu128 with CUDA 12.8, NumPy 2.5.4, Pillow 13.0.0, and SciPy
1.18.1, with `torch.cuda.is_available()` true. A generic two-layer 64×64
synthetic convolution model completed two warmup and three timed CUDA
forward/backward/AdamW steps with finite loss, 0.0695 seconds measured time,
1,804,288 bytes peak allocated, and 4,194,304 bytes peak reserved. This generic
smoke is not project pilot fit, learning evidence, Stage 0 passage, or a study.
At the user's request,
the GitHub repository was made public and cloned into the laptop's `source`
subdirectory, with the isolated `.venv` in its parent. Locked `uv sync` and 127
Windows tests passed. A 100-step project run, batch size 2 and width 16, took
11.0 seconds including rendering, with 43,783,680 bytes peak allocated and
65,011,712 bytes reserved. Its sample was noise; these numbers are a short
engineering measurement, not an estimate of the research budget.
The 2,000-step run used 4,000 scenes and took 167.536 seconds including
rendering; its sample was also noise. An 8,000-step deterministic project pilot
was measured at code revision `4287d88`: training continued the existing seed-45,
batch-size-2, width-16, 1,000-timestep run across two additional 2,000-step
checkpointed segments (building on the earlier 4,000-step checkpoint). Cumulative
recorded training time was 802.497 seconds, with peak allocated memory of
96,212,480 bytes and reserved memory of 111,149,056 bytes. Four example
conditions were sampled once each using seeds 5, 6, 7, 8; visual inspection found
noise or near-blank images with colored fragments, none clearly showing the
intended three-object scene. This is only four inspected samples, not a formal
quality score, and they are not ready for human audit. Checkpoints and samples
are in ignored `outputs/deterministic-pilot-5549zx7k` on the laptop. Stage 0
remains explicitly unpassed. Checkpoints now follow schema version 3, recording
determinism state and PyTorch backend runtime flags. Strict deterministic training
(`--deterministic`) configures the cuBLAS workspace and PyTorch deterministic
algorithms while rejecting unsupported ops. Older v1 and v2 checkpoints remain
compatible for non-strict sampling, but resuming requires schema 3. A small
integrated deterministic CUDA replay matched exactly; Stage 0 remains not passed.

The run later reached 12,000 updates in 1,210.404 cumulative seconds. One red
and one relational sample remained noise; two samples are not a quality score.
One-step denoising errors decreased between 4,000 and 12,000 updates on a
small matched probe, while an exact-noise oracle passed the sampler wiring
check. Neither result validates generated images.

A deterministic 4,000-update fixed-scene diagnostic on the RTX 3050 (target scene
seed 77, train seed 46, sample seed 5, batch 2, width 16, 1,000 timesteps,
code revision `35500f7`) finished in 365.534 seconds, using 96,207,872 bytes peak
allocated and 111,149,056 bytes peak reserved CUDA memory. The fixed target
image contained three objects, while the sampled image appeared mostly dark noise
(image MSE on `[0, 1]` was 0.8017; artifacts in ignored
`outputs/static-scene-769aef5ec6` on the laptop). This is one diagnostic sample,
not a general model bug, training failure, or Stage 0 passage.

A subsequent truncated-reconstruction diagnostic on the same fixed target
performed well at several lower noise levels but degraded sharply at the highest
tested noise level. A one-step high-t baseline on that checkpoint (CPU, 16 seeds,
timesteps 900, 950, 990, and 999) measured mean noise MSEs of about
0.0199–0.0212 for raw and 0.0274–0.0284 for EMA weights; the copy-noisy-input
baseline was 0.00004–0.00025, the zero predictor about 0.995819, and the exact
oracle zero. These measurements come from one target and checkpoint, not a full
reverse-sampling or generalization result. Both diagnostics and their caveats
are in
[pilot training](pilot-training.md#one-step-high-t-baseline-diagnostic) and
[pilot training](pilot-training.md#truncated-reconstruction-diagnostic); they do
not establish a cause or pass Stage 0.

A paired CUDA fine-tuning diagnostic at code revision `4c8c798` trained a
matched control arm and a high-timestep-weighted arm for 500 updates each from
that checkpoint, with training seed 46, batch size 2, and the full 1,000-step
schedule. Both arms shared identical timesteps and realized a high-timestep
fraction of 0.103; the weighted arm applied multiplier 5 at timesteps 900 and
above, normalized by expected weight 1.4. One-step evaluation over CPU seeds 0–7
gave t=999 noise MSE of 0.017361 raw and 0.022943 EMA for control, and 0.016514
raw and 0.022615 EMA for weighted, versus 0.019759 raw and 0.027238 EMA at
start. At t=100, weighted raw noise MSE was 0.051213 versus 0.044935 for
control. Training took 51.57 seconds for control and 50.41 seconds for weighted,
with 96,152,064 bytes peak allocated for both. This is one fixed scene, one
checkpoint, and one training seed; it reports one-step errors only and does not
establish full-sampling behavior, generalization, human audit accuracy, or Stage
0. See
[the paired diagnostic](pilot-training.md#paired-high-t-fine-tuning-diagnostic).

A 300-scene clean-render audit at seeds 30000–30299 covered all 57 candidate
rules, with zero detector failures and zero observed rule errors. Some rules
had as few as five positive scenes, so these counts do not establish a 2% error
bound. A 100-scene-per-condition matrix smoke test at seed 43 yielded an
estimated singleton-preserving equality rank of 10 and nullity of 47. Twenty
designs were sampled with seed 43 and a 0.001 minimum weight. All 1,140
rule-by-design rows had finite marginal and lift estimates. The pooled
log-mean-marginal/log-lift correlation was -0.446; VIF was 11.61 for log lift
and 13.72 for relation count. Exact LP bounds over the same estimated matrix
span lift `[-4.888, +0.337]` with median width `1.172`, and `23/57` rules admit
positive lift; a 5-chain × 200-design run still found `0/57,000` positive lifts.
These rows share rules and designs, and the exposures and equality subspace come
from the same small Monte Carlo matrix. The statistics are exploratory, not
confirmatory, and do not establish the proposal's decorrelation criteria;
`β_m` is exactly collinear with the rule intercept. Reproduce with:

```sh
uv run python -m exposure.design_sample \
  --report outputs/exposure-57-pilot.json \
  --weights outputs/uniform-57.json \
  --samples 20 --seed 43 --min-weight 0.001 \
  --burn-in 100 --thin 10 --output outputs/design-samples-57.json
uv run python -m exposure.design_diagnostics \
  --report outputs/exposure-57-pilot.json \
  --designs outputs/design-samples-57.json \
  --output outputs/design-diagnostics-57.json
```

Local reports are written under `outputs/` and are not tracked in Git.
