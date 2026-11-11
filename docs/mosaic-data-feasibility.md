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
It reports loss traces and source/runtime hashes but does not save checkpoints,
generate images, score the evaluation set, or measure model quality.

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
without the missing sampler, frozen evaluation protocol and independent runs.
