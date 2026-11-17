# CPU image diagnostics

`exposure.image_diagnostics` measures the radially binned 2D FFT power of an image, and optionally compares an image to its **actual spatially paired** clean reference using instance masks. These are instrument-development measurements, not validation of H6, generation quality, or a Stage 0 pass.

```sh
# Spectral analysis works on generated images without inventing a target.
uv run python -m exposure.image_diagnostics \
  --image outputs/preview-800000-20261111/red-101.png \
  --output outputs/image-spectrum.json

# Known pairing: render seed 7, perturb the same image using blur radius 1.
uv run python -m exposure.image_diagnostics \
  --render-seed 7 --blur 1 \
  --output outputs/image-diagnostics-blur-seed7.json

# Explicit paired data: reference and target must depict the same scene in
# the same coordinate frame; integer PNG uses zero for background and a
# different positive ID for each instance. The flag records your assertion.
uv run python -m exposure.image_diagnostics \
  --image prediction.png --reference ground_truth.png \
  --instances instance_ids.png --assert-aligned \
  --output outputs/image-diagnostics-paired.json
```

The spectrum uses mean-centered Rec. 709 weighted gamma-encoded RGB and `abs(FFT)^2/(H*W)^2`, averaged and summed in integer frequency-index radius bins. The sum of all bin powers equals pixel luminance variance (Parseval); no physical radiance interpretation is implied. For rectangular images, radius is measured in integer frequency indices rather than a common cycles/pixel scale. Inputs should have identical dimensions if spectra are compared, and a single image does not represent a checkpoint's distribution.

The per-region error is RGB mean-squared difference after division by 255, averaged over the pixel's three channels and over pixels in the region. Interior is the eroded object-instance union; boundary is a one-pixel morphological inner/outer ring by default; gap is background within eight pixels of at least two separate instances, excluding boundary. These masks are disjoint and do not necessarily cover the image. An empty region returns `null`, not zero. The renderer example constructs the instance masks by rendering each annotated object independently, with a threshold of 250 to exclude antialias fringes, and rejects overlapping masks. Other datasets must supply their own valid instance map.

For the deterministic seed-7 synthetic blur example above, the implementation measured 274 interior pixels at RGB MSE 0.01430, 258 boundary pixels at 0.01114 and 100 gap pixels at 0.0000696; overall MSE 0.001667. These numbers characterize one intentionally blurred renderer scene, not the diffusion models. The generated report is ignored under `outputs/`.

One descriptive sample each from the 310k, 800k and matched 500k artifacts also produced a spectrum in `outputs/image-diagnostics-three-samples.json`. Those models, conditions and samples are not paired, so comparing the values as checkpoint progress would be invalid. Sampling random diffusion generations does not provide a paired reference image. Evaluating regional reconstruction error on checkpoint outputs therefore requires a fixed input-image reconstruction or another demonstrably aligned target, not a randomly chosen clean render.

The proposal to claim a reduction to single-digit checker attrition is **not yet established**. The foreground segmenter described below has been trained and tested on synthetic scenes, but a rule-checking system has not been validated. Robust rule evaluation requires negative and difficult cases, fixed checker decisions, and independent human labels. Pixel masks alone do not validate object color, identity, relations, or predicate truth.

## Experimental foreground segmentation prototype

`exposure.checker_seg` now contains a small three-level U-Net training scaffold. It predicts a **binary foreground mask only**: there is no object extraction, color or shape prediction, relation evaluation, rule truth, abstention policy, or compatibility claim with `checker_perturb.py`. In particular, pixel-level segmentation IoU does not prove that checker blur attrition fell or that the model works on generated images.

For a later bounded experiment on a machine with optional PyTorch installed:

```sh
uv run --extra train python -m exposure.checker_seg \
  --output outputs/foreground-seg-98200 \
  --seed 98200 --train-count 600 --val-count 100 --test-count 100 \
  --epochs 3 --batch-size 16 --device cuda
```

The command refuses to overwrite an existing output directory. It renders scene seeds from disjoint contiguous training, validation and test intervals and checks all pixel SHA-256 values for duplicates. The segmentation target is generated from renderer object annotations. During training, it applies seeded brightness, Gaussian blur, and noise to the input while retaining the unwarped mask. It selects the model using blur-1 validation pixel IoU, and reports held-out clean, blur-1, brightness 1.10, and noise pixel confusion counts, foreground IoU, foreground recall, and background false-positive rate. Hashes and the selected-model protocol are saved with the report. These perturbations are not a substitute for generated-image negative cases or independently annotated segmentation masks.

**Status:** local split/augmentation/metric tests passed; the local PyTorch-dependent test was skipped because PyTorch is unavailable here. A separate one-epoch smoke on the laptop CPU (32 training, 8 validation, 8 test scenes; width 4; batch 4; one CPU thread) completed in an isolated workspace while diffusion training remained untouched. The model labeled every held-out pixel background: across all four test perturbations, IoU and foreground recall were both 0 (0 true-positive and 3,598 false-negative foreground pixels). This is a failed learnability result for a deliberately tiny run, not a checker or claim about what a longer training could achieve. The report is preserved as `outputs/foreground-seg-smoke-98200.json` (ignored). A subsequent bounded GPU run used 600/100/100 disjoint train/validation/test scenes, 3 epochs, width 16 and batch 8. On 100 held-out synthetic scenes, the foreground pixel IoU was 0.9427 for clean and 0.9347 for blur-1; foreground recall was 0.9818 and 0.9713 respectively. This is pooled pixel-level segmentation, not scene-level checker coverage or predicate validity. The generated checkpoint and report are in ignored `outputs/foreground-seg-98200/`; report SHA-256 `3dd54641fa3200b04e4c4a11e6d35f5fa60c0b97d3c3710c8d9d42f80808f95f`, weights SHA-256 `58ddfa1d8899832c7983f60fcf69cd3dce1545d867516cccfb2e7935c1dfd979`. No checker attrition reduction or result on the 300-image audit has been produced. Before treating this as a checker, implement object and predicate interpretation with explicit abstention, evaluate correctness and coverage on blinded human-labeled generated images including negative cases, and review domain shift and mask boundaries.

## Experimental rule adapter: frozen negative result

To investigate whether segmentation can support rule checking, `exposure.checker_seg_adapter` adds an **experimental**, pre-fixed adapter: threshold foreground logits at zero, identify connected components, require three separated renderer-sized regions, classify shape using disjoint fill-ratio intervals, select a nearest unambiguous palette color, and abstain for ambiguous geometry or close center coordinates. Binary foreground masks cannot reliably distinguish merged objects, and none of these thresholds certifies semantic correctness. They were fixed before applying the adapter to the reviewed model images; adapter SHA-256 `50027dcecbdc8ac3bd42760ece3138d906e925c5273176df238ed5ca0d847f59`, segmentation weights SHA-256 `58ddfa1d8899832c7983f60fcf69cd3dce1545d867516cccfb2e7935c1dfd979`.

On 200 renderer scenes with disjoint seeds 99000–99199 (training/validation/test seeds were 98200–98999), the standard checker abstained on **172/200 (86%)** with blur radius 1; the adapter abstained on **147/200 (73.5%)**. These same-scene checks show a modest coverage increase, not the proposed single-digit attrition. Among the adapter's 53 accepted blur scenes, across 477 correlated scene-rule decisions, it made 17 false positive and 37 false negative rule decisions. The standard checker made no false decisions across its 28 accepted scenes, but abstained on 172. Comparisons concern different selected subsets, so accepted-only error rates are not an apples-to-apples accuracy comparison.

The frozen adapter was then applied to the previously reviewed 300-image 800k pilot; review answers were not transferred to the prediction laptop. Prediction file SHA-256 `5699ebf31357793f55a800197e3894f8e21da23c03390454c7a6fa4c7154f291`. Its 54/300 decisions (18%) are fewer than the standard checker's 95/300 (31.7%). Against the original reviews, 53 accepted cases had resolved human labels: 42 true positives, 10 true negatives, one false positive and no false negatives. One other accepted case had conflicting human reviews and remains unresolved; two more unresolved cases were abstentions. The 244 remaining abstentions with resolved human labels comprised 179 yes and 65 no. The images and accepted decisions are correlated; this is exploratory evaluation on previously reviewed data, not a fresh independent confirmation and not a Stage 0 pass. Frozen predictions, synthetic results and the score are retained in ignored `outputs/foreground-seg-adapter-98200/`; score SHA-256 `5bcf0808bee0ed0cd2005f58d3c0bcc9d22affdf333da895b38e68e2a0e4dac2`.

This adapter does **not** establish a reliable replacement checker. A promising segmentation IoU does not validate shape interpretation, color assignment or relationships. Any later threshold change creates a new candidate and requires a new held-out audit; the above results must not be reported as validation of that changed candidate.
