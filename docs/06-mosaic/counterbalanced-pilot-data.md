# Counterbalanced paired-scene pilot candidate

`exposure.counterbalanced_pilot_data` generates a separate synthetic dataset candidate addressing two limitations of the historical four-cell pilot: role–attribute confounding and unmatched low/high scene geometry. It does not modify the old dataset, train a model, implement the global exposure-mixture intervention, revise spatial-relation semantics, validate a checker, or pass Stage 0.

Each pair uses the same target and first distractor, including their color, shape and coordinates. The high-complexity image adds a second distractor sampled at a different grid position. The underlying object bounding boxes do not overlap; raster antialias fringes are not an independent segmentation ground truth. Each manifest record identifies its pair, split, complexity, target attributes, object annotations, image filename and raw-pixel SHA-256.

In the fully balanced mode, every one of the 12 color–shape combinations occurs equally often in each object role across the high-complexity scenes of each split. For a compositional holdout, pass `--holdout COLOR:SHAPE` for each excluded combination: **none** can occur in any object role in training, while the test target vocabulary contains all 12 combinations equally often. Training role balance then applies to the remaining combinations, not to all 12. The deterministic next-two-combination assignment balances role counts but creates systematic relationships among the objects' attributes; it is not an independently randomized factorial distribution. Consequently it is a candidate for testing the data pipeline, not evidence of unbiased compositional generalization or a finished experimental design.

Split assignment is made before generating pairs; a bounded rejection check rejects exact pixel duplicates across both splits and within them. This does not prove the absence of near duplicates, background artifacts, or shared underlying factors. Train/test seeds and samples should not be treated as independent model-training runs. A real study still needs a model conditioned on **both** target attributes, a frozen sampling protocol, a validated evaluator, independent dataset/model seeds, and an assessment of memorization and uncertainty. The existing single-color PilotUNet conditioning cannot directly test a two-attribute held-out target.

Example: the following selects three combinations to hold out; the choice is illustrative, not justified as a final research split. Train-pair count must be a multiple of the number of remaining combinations; test-pair count must be a multiple of 12. The output directory must be new. A failed generation can leave a partial directory, which should be inspected or removed before retrying.

```sh
uv run --no-sync python -m exposure.counterbalanced_pilot_data \
  --output outputs/counterbalanced-pilot-new --seed 82700 \
  --train-pairs 108 --test-pairs 24 \
  --holdout red:circle --holdout blue:square --holdout green:triangle
```

A local demonstration at the same seed produced 108 train and 24 test pairs (264 images): each of the nine allowed training targets appeared 12 times; each of the 12 test targets appeared twice. Its manifest SHA-256 is `924611c7cf20d57bb6650909837729b7bdb7d60129438fe1efc55af138268c5e`. The generated directory is ignored by Git. Unit tests check role balance, paired object identity and geometry, geometric non-overlap, exact split uniqueness, held-out combinations in all training roles, and overwrite prevention.
