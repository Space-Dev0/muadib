# Stage 0 status

**Not passed.** The following measurements are instrument-development results,
not confirmatory research findings.

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

A fixed-design holdout using a second 100-scene-per-condition measurement
matrix gave pooled correlation -0.4640 and log-lift VIF 11.07; the maximum
singleton deviation from the uniform reference was 0.01767. This exploratory
analysis omits breadth and does not assess the final gate. See
[design holdout](design-holdout.md).

| Component | Current evidence | Remaining work |
| --- | --- | --- |
| Renderer and rules | 2D renderer, annotation-based rule evaluation and grammar witnesses | Finalize and validate the benchmark vocabulary and full counterbalancing. |
| Candidate templates | Provisional 57-template generator covers all singleton colors and shapes in each one-attribute cell; a 100-scene-per-template sampling smoke test completed | Confirm full counterbalancing, held-out rules and sufficient sampling precision. |
| Pilot training | Locked sync and 127 Windows tests passed. A deterministic pilot reached 12,000 steps with checkpoints, and a 4,000-step fixed-scene diagnostic ran on RTX 3050 with truncated-reconstruction, one-step high-t baseline, and paired high-t fine-tuning follow-ups. | Inspected online samples were noise; single fixed-scene sample was mostly dark noise (MSE 0.8017); no validated model outputs, reference model, controlled guidance sweep, or human audit. See [pilot training](pilot-training.md). |
| Checker | Clean-render and controlled perturbation audits; conservative sharpening fallback | High rejection remains at brightness 1.10, blur 0.50, and blur 1.00; validate every relevant rule and audit generated images with human judgments. |
| Exposure matrix and design | Monte Carlo estimates, LP feasibility and numerical minimum-KL allocation | Quantify matrix uncertainty and verify feasible exposure manipulations and leave-out probes on the final benchmark. |
| Decorrelation | Twenty exploratory hit-and-run designs from the 57-condition pilot; pooled correlations and VIF computed over 1,140 rule-by-design rows | Define breadth, freeze the analysis table and uncertainty method, and evaluate the registered criteria. Current estimates do not assess the proposal gate. |
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
and 13.72 for relation count. These rows share rules and designs, and the
exposures and equality subspace come from the same small Monte Carlo matrix.
The statistics are exploratory, not confirmatory and do not establish the
proposal's decorrelation criteria. Reproduce with:

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
