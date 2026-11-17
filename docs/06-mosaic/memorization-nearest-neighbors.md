# Nearest-neighbor audit of the matched pilot

This is a **descriptive pixel-space audit**, not a memorization-rate estimate. The review correctly identified a risk: 500,000 updates × batch size 4 / 200 training images is 10,000 dataset-equivalent exposures per cell. That count alone does not prove memorization.

For each generated output in the fixed 64-image matched 500k evaluation, `exposure.memorization_audit` finds the closest training image from the **same training cell and requested target color** using mean squared error over 64×64 RGB pixels scaled to [0,1]. As a comparison, it measures distances to the same training references for separately rendered evaluation controls with the same object-count condition and target color. The data manifest hash matches the recorded training dataset; the tool checks every dataset image's raw-pixel hash and each generated file and raw-pixel hash against the recorded report, verifies the checkpoint hash against the evaluation protocol, and requires the exact 16 color/seed combinations per cell. These checks do not independently attest model training or make the control distribution equivalent to the generated distribution.

| Cell | Target | Training references | Generated median MSE (8 images) | Renderer-control median MSE (20 images) |
| --- | --- | ---: | ---: | ---: |
| Low / balanced | red | 100 | 0.01684 | 0.01587 |
| Low / balanced | blue | 100 | 0.01321 | 0.01380 |
| Low / imbalanced | red | 180 | 0.01369 | 0.01374 |
| Low / imbalanced | blue | 20 | 0.02099 | 0.02049 |
| High / balanced | red | 100 | 0.02935 | 0.02501 |
| High / balanced | blue | 100 | 0.02626 | 0.02575 |
| High / imbalanced | red | 180 | 0.02354 | 0.02649 |
| High / imbalanced | blue | 20 | 0.03059 | 0.03121 |

None of the 64 generated images is an exact pixel match to the selected color/cell training references; none of the renderer controls is either. This does **not** establish absence of memorization: small shifts, blur, colors, stochastic noise or other changes can make memorized scenes have nonzero MSE. Different numbers of available references also affect nearest-neighbor distances, so the 180-reference and 20-reference groups should not be compared as if reference count were fixed. Generated images and clean renderer controls differ in visual quality and distribution. The same controls are reused for the two balance conditions; generated noise seeds are paired across cells and colors. These are not independent model or dataset seeds, and no statistical interval or binary copy threshold has been validated.

Reproduce from the repository root when the ignored artifacts are available:

```sh
OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m exposure.memorization_audit \
  --dataset outputs/mosaic-data-gpu-feasibility-81300 \
  --evaluation outputs/mosaic-matched-eval-500k-82000 \
  --expected-manifest-sha256 48bf70c4d43c07bdc563cba1f585ea618e862047635257a4ce63f67c9bd45d37 \
  --output outputs/mosaic-matched-nearest-new.json
```

The detailed local report `outputs/mosaic-matched-nearest-82711.json` has SHA-256 `fe47b54d850f494970a15ca8de5e6ad0b345a2085f5c24ae2e113bc537ec68d3`. It includes every query, selected training-image filename and MSE; its inputs and report are ignored outputs, not guaranteed public artifacts. Next steps are to inspect nearest pairs, design and calibrate transformation-sensitive or feature-based metrics with known-copy/non-copy controls, match reference-set sizes where comparison calls for it, and evaluate independent datasets and model seeds. No Stage 0 pass is implied.
