# Stage 0 status

**Not passed.** The following measurements are instrument-development results,
not confirmatory research findings.

| Component | Current evidence | Remaining work |
| --- | --- | --- |
| Renderer and rules | 2D renderer, annotation-based rule evaluation and grammar witnesses | Finalize and validate the benchmark vocabulary and full counterbalancing. |
| Candidate templates | Provisional 57-template generator covers all singleton colors and shapes in each one-attribute cell; a 100-scene-per-template sampling smoke test completed | Confirm full counterbalancing, held-out rules and sufficient sampling precision. |
| Pilot training | Locked sync and 127 Windows tests passed. The deterministic CUDA option matched a small split run; an 8,000-step deterministic pilot (continued across four 2,000-update segments with resume) completed. | The 100-, 2,000-, 4,000-, and 8,000-step samples were noise or near-blank fragments; four inspected samples are not ready for human audit; no validated model outputs, reference model, or controlled guidance sweep. See [pilot training](pilot-training.md). |
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
