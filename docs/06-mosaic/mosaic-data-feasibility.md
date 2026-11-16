# MOSAIC-inspired data feasibility pilot

This optional toy dataset tests whether a matched 2×2 contrast can be built
with this project's renderer. It does **not** train a model, run MOSAIC, or
pass a Stage 0 gate. It does not represent the proposal's final benchmark.

```sh
uv run python -m exposure.mosaic_pilot_data --seed 81203 \
  --train-per-cell 20 --eval-per-complexity 10 \
  --output outputs/mosaic-data-feasibility-81203
```

Four distinct training cells cross low/high complexity (one target and one or
two distractors) with balanced/imbalanced target color (red:blue 50:50 or
90:10). A red or blue circle is the target; distractors are green/yellow
squares or triangles. Each cell has an identical count of rendered scenes.
The shared evaluation set is drawn independently of every training cell and
is balanced across target color and complexity. Duplicate pixel arrays are
rejected across all cells and evaluation. Each record contains object labels,
class, split, complexity, cell, and a raw pixel SHA-256. Output directories
are exclusive and never overwritten. Small default sizes are a plumbing smoke
test; they provide no statistical conclusion.

A valid adapted experiment still requires independent models trained per cell,
matched architecture, update count and training conditions, generation of a
prespecified evaluation sample, a separately validated evaluator or human
labels, multiple independently trained seed blocks, and a documented protocol
and comparison with MOSAIC. The online trainer currently samples three-object
scenes and cannot consume this dataset directly. The common evaluation mixture
includes both complexity levels, so its interpretation includes out-of-training
complexity conditions. Do not submit an empty or synthetic-results file to
`mosaic_contrast` as if it were an experiment.

## CPU training-plumbing smoke

`exposure.mosaic_toy_train` reads the generated manifest, verifies pixel hashes
and the four cell quotas, and runs four distinct tiny velocity-prediction
models with matched initialization, optimizer settings, batch size, and update
count. It uses the target class as a condition. This is an intentionally small
new model, not the project checkpoint or MOSAIC's latent diffusion backbone.
It reports loss traces and source/runtime hashes. An optional checkpoint
directory stores four non-resumable model checkpoints; they do not contain
optimizer or random-generator state.

```sh
uv run --extra train python -m exposure.mosaic_toy_train \
  --dataset outputs/mosaic-data-feasibility-81203 --seed 81204 \
  --steps 2 --batch-size 2 --width 8 --device cpu \
  --output outputs/mosaic-toy-train-smoke-81204.json
```

A local CPU smoke with those settings completed all four cells with finite
losses and the same initial parameter hash. The two updates and 20 training
images per cell are far too small to infer learnability, effect direction,
uncertainty or a MOSAIC replication. Longer training cannot be interpreted
without a validated sampler, frozen evaluation protocol and independent runs.

For an end-to-end plumbing check, pass `--checkpoint-dir` with a new path to
the trainer, then run, for example:

```sh
uv run --extra train python -m exposure.mosaic_toy_sample \
  --checkpoint outputs/toy-checkpoints/low-balanced.pt --label red --seed 7 \
  --steps 10 --device cpu --output outputs/toy-sample.png
```

The sampler uses deterministic velocity inversion and the same linear alpha
schedule as the toy trainer. It starts from Gaussian noise at alpha 0.01, an
approximation to the terminal training distribution. It does not implement
classifier-free guidance or evaluate sample quality. A two-update-per-cell
CPU smoke produced four checkpoints and one sample, which appeared as noise;
this is expected for a plumbing test and is not a learnability result.

On 2026-11-13, CPU diagnostics trained four cells for 500 updates each (seed
81206) and separately 5,000 updates each (seed 81207), each using 20 training
images per cell, batch size 2 and width 8. One sample per cell and target class
at seed 7 and 50 sampler steps was inspected for each run; both sets appeared
mostly noise, with no reliable comparison of complexity or imbalance. The
small 3-convolution toy architecture and these budgets do not establish that a
larger model or longer training would fail. Reports, checkpoints and samples
are under ignored `outputs/mosaic-toy-cpu-*81206*` and `*81207*` paths. The
first 81206 attempt was interrupted after one checkpoint; its incomplete
directory is retained separately from the completed retry. These runs are not
the MOSAIC replication, and individual training losses are not replication
metrics.

The one-image `exposure.mosaic_toy_overfit` diagnostic uses a separate fixed
evaluation noise draw and reports reconstructed x0 MSE at alpha 0.01, 0.5,
and 0.99 before and after training. On the first red training image at seed
81209, width 8 and 5,000 updates, those MSEs changed from 0.9790/0.5291/0.0105
to 0.1979/0.0411/0.0082. These values use one scene and one evaluation noise
draw, and do not test full sampling, generalization, or Stage 0. The ignored
report is `outputs/mosaic-toy-fixed-overfit-81209.json`; a multi-step oracle
velocity inversion unit test separately verifies the sampler's algebra.

For local CPU validation, an isolated temporary environment with PyTorch
2.9.1+cpu ran the project test suite on 2026-11-11: 233 tests passed. The local
machine reported Intel Iris Xe graphics and no NVIDIA CUDA device; the CPU
environment is not evidence of CUDA execution or a research run.
