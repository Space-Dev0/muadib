# Paired sampling-step diagnostic

The exploratory 20,000-update multiscale samples were blotchy. To test whether
more reverse-sampling passes alone fix these particular examples, we held the
weights, CUDA device, requested label and noise seed fixed, and changed only
sampling steps from 50 to 200 and 500. We used low-balanced and high-balanced
checkpoints, red and blue labels, and seed 81330: four paired comparisons,
12 images in total. No training was performed.

All four newly generated 50-step images exactly match the corresponding
original 20,000-update preview at the pixel level. Visually the 200- and
500-step images retain essentially the same overlapping, distorted shapes.
Mean absolute RGB changes on the [0, 1] scale were:

| Model and label | 50 → 200 | 200 → 500 |
|---|---:|---:|
| Low, red | 0.01261 | 0.00267 |
| Low, blue | 0.00581 | 0.00133 |
| High, red | 0.01642 | 0.00297 |
| High, blue | 0.01697 | 0.00172 |

Pixel change is not an image-quality metric. This small, matched probe supports
only the observation that increasing these sampler steps did not visibly repair
these examples; it does not rule out other samplers, different seeds, or longer
training. The same shared seed across models and labels does not establish
independent observations. It does not pass a Stage 0 or MOSAIC gate.

Local ignored outputs are `outputs/mosaic-steps-low-81330/`,
`outputs/mosaic-steps-high-81330/`, and the labeled contact sheet
`outputs/mosaic-steps-81330-contact.png`. Each directory has checkpoint, code,
image hashes and runtime details in `report.json`.

From the GPU laptop's staged source directory, the low cell can be reproduced
with a fresh output path using:

```sh
python -m exposure.mosaic_step_probe \
  --checkpoint outputs/mosaic-multiscale-probe-81322/multiscale-20000.pt \
  --output outputs/mosaic-steps-low-new --device cuda --seed 81330
```

The high checkpoint is
`outputs/mosaic-multiscale-81322-high-balanced/multiscale-20000.pt`.
The generated image values can vary with hardware, PyTorch and runtime settings.

[DDIM](https://arxiv.org/abs/2010.02502) describes the broader tradeoff between
sampling computation and sample quality. This single-model-family experiment
tests step counts in our current deterministic sampler; it does not compare
alternative schedules or establish a general optimum.
