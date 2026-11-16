# Toy sampler trajectory diagnostic

The one-step denoising test does not test repeated sampler updates. This diagnostic
starts from a known held-out image mixed with seeded Gaussian noise at alpha=0.01,
then follows the same velocity inversion and deterministic 50-step schedule as the
toy sampler. Every model receives the same noisy input for each evaluation image.
An oracle that knows both the clean image and noise exercises the same equations.
The models do not have this oracle information.

Run from the repository root with the project's PyTorch environment:

```sh
uv run python -m exposure.mosaic_toy_trajectory \
  --dataset outputs/mosaic-data-feasibility-81203 \
  --checkpoints outputs/mosaic-toy-cpu-checkpoints-81207 \
  --seed 81217 --steps 50 --device cpu \
  --output outputs/mosaic-toy-trajectory-81217.json
```

On the existing 5,000-update CPU checkpoints and 20 held-out images, mean squared
reconstruction errors by training cell, evaluation complexity and target color
range from 0.2090 to 0.2716 on the [-1, 1] pixel scale; the maximum corresponding
oracle mean is 2.54e-13. Each subgroup contains only five images. These results
verify the numerical oracle path and provide a debugging reference; they do not
establish model convergence or statistically meaningful differences between cells.
The implementation verifies the dataset hashes and matched checkpoint metadata;
its output records the dataset, checkpoints and diagnostic source hashes.

This is a reconstruction test from an exact forward-noised image, whereas normal
sampling starts from pure Gaussian noise. A model is not expected to recover an
arbitrary held-out image from pure noise. These errors therefore do not measure
sample quality, color recognition, the MOSAIC effect, or the Stage 0 gate.
