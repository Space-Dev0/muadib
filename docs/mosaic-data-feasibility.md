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
