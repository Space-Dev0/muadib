# Toy model conditioning probe

This is a debugging probe, not the MOSAIC contrast, an image quality metric,
or a Stage 0 decision. The existing 5,000-update toy samples looked like noise;
this probe asks the narrower question of whether changing the supplied class
condition changes the final pixels for exactly the same sampler noise. A change
in pixels is not evidence that an image contains the requested object.

Run on the four existing matched toy checkpoints:

```sh
uv run --extra train python -m exposure.mosaic_toy_condition_probe \
  --checkpoints outputs/mosaic-toy-cpu-checkpoints-81207 \
  --first-seed 81231 --seeds 8 --steps 50 --device cpu \
  --output outputs/mosaic-toy-condition-81231-v2.json
```

The code uses the same sampler noise seeds for red and blue conditions and
across the four models, and verifies matched checkpoint metadata. It calculates
mean absolute RGB differences between paired PNG outputs on [0,1], and a signed
whole-image contrast: mean(R−B) with the red condition minus mean(R−B) with the
blue condition. Both include background and clipped/quantized pixels and are
sensitive to color artifacts. Neither is a classifier of the generated object.

| Training cell | Mean absolute RGB difference | Signed color contrast |
|---|---:|---:|
| Low complexity, balanced | 0.04153 | −0.00330 |
| Low complexity, imbalanced | 0.04665 | 0.03282 |
| High complexity, balanced | 0.03484 | −0.00418 |
| High complexity, imbalanced | 0.04556 | 0.04111 |

These eight paired noise seeds come from one training seed block and one dataset.
All balanced model seed-level signed contrasts were negative and all imbalanced
ones positive in this small probe. The label affects pixels, but the contrast
is not reliably positive across training cells and cannot tell whether target
objects exist. It also cannot distinguish semantically meaningful colors from
global tint and does not establish a causal effect of imbalance. New independent
training seeds, well-formed samples and a separately validated image evaluator
are required before assessing the intended complexity-versus-imbalance direction.
The toy model and pixel-space implementation differ substantially from MOSAIC's
latent diffusion and classifier evaluation; this is not a reproduction.
