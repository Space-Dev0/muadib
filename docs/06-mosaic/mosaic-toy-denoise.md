# Paired one-step toy-model denoising diagnostic

`exposure.mosaic_toy_denoise` measures reconstruction of **known held-out
images** by the four toy velocity models. It is not generation from noise,
model quality, a MOSAIC replication, or a Stage 0 decision. It verifies image
hashes and uniqueness across training and evaluation, requires a balanced
evaluation set, and checks the four checkpoint cells and matched training
configuration. One independently seeded noise draw per image is reused across
all four models and all noise levels for paired descriptive comparisons.

For each held-out image and alpha, it constructs a noisy image from the known
clean target and noise, predicts velocity, and recovers x0 by algebraic
inversion. It reports mean x0 MSE per training cell, evaluation complexity and
noise level. It also reports the error of copying the noisy input, a zero-image
baseline, and the known-noise oracle. Alpha 0.01 means much more noise than
alpha 0.99. Global pixel MSE can be dominated by the background and says
nothing directly about shape, color, relation, or generated-image success.

```sh
uv run --extra train python -m exposure.mosaic_toy_denoise \
  --dataset outputs/mosaic-data-feasibility-81203 \
  --checkpoints outputs/mosaic-toy-cpu-checkpoints-81207 \
  --seed 81215 --device cpu --output outputs/mosaic-toy-denoise-81215-v2.json
```

The local CPU run used 20 held-out images (10 per complexity; 5 per target
color within complexity) and four models trained for 5,000 updates each with
the same initialization and matched settings. Their mean x0 MSE across the
eight cell-by-evaluation-complexity groups ranged from 0.2150–0.2662 at
alpha 0.01, 0.0555–0.0759 at alpha 0.5, and 0.0078–0.0082 at alpha 0.99.
The corresponding copy-noisy-input baseline ranges were 1.7591–1.7680,
approximately 0.5829, and 0.0100. The algebraic oracle MSE was below 1e-12.
These are a single seed block, small shared evaluation set and one noise draw
per image, not independent training replicates or a confidence interval.
They do not resolve why full reverse samples looked like noise, nor compare
directions with MOSAIC. Further work requires validated generation and
evaluation, multiple independent training seeds, and a frozen protocol.
