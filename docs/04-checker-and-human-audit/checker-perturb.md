# Checker perturbation audit

This prototype audit measures how the image checker responds to simple changes
in otherwise identical synthetic scenes. Run from the repository root:

```sh
uv run python -m exposure.checker_perturb --samples 200 --seed 41000 --output outputs/checker-perturb-41000.json
```

It applies five cases separately to each seeded scene: unchanged, brightness
factors 0.90 and 1.10, and Gaussian blur radii 0.50 and 1.00 pixels. Brightness
changes the background too. These cases preserve object coordinates, so the
renderer annotations remain the ground truth, but they are not a model of
diffusion output defects.

On 200 separate synthetic seeds (41000–41199), the current checker coverage was:

- clean: 200/200;
- brightness 0.90: 200/200;
- brightness 1.10: 114/200;
- blur radius 0.50: 114/200;
- blur radius 1.00: 31/200.

The previous implementation baseline on these seeds was 85/200 for blur radius
0.50. No rule errors were observed on accepted examples.
Rejected scenes are excluded, so these coverage counts do not establish overall
checker validity. At blur radius 1.00 the counts are rejection `169/200`
(`84.5%`) and coverage `31/200`; on a separate 100-scene holdout, rejection is
`90/100` (`90%`). Among accepted images there are zero false accepts and zero
false rejects, so the failure mode is **attrition**, not misclassification.

For each case, the output records `total`, `evaluated`, `rejected`,
`rejection_rate`, and `coverage`. For each rule, ground-truth positive and
negative totals include rejected images. `positive` and `negative` are
accepted-scene denominators for `false_rejects` and `false_accepts`,
respectively. The `detector_failure_positive` and
`detector_failure_negative` fields split rejected images by their true rule
status. Each rule also reports:

- `conditional_sensitivity` / `conditional_specificity`: accepted images only;
- `unconditional_sensitivity`: failures on positives count as missed
  positives;
- `unconditional_specificity`: correctly classified negatives divided by all
  ground-truth negatives; failures are included in the denominator but are not
  counted as false positives;
- `positive_coverage` / `negative_coverage`: accepted fractions by truth.

Rejections are not counted as false negatives or false positives; error rates
on accepted images must therefore be interpreted with coverage. Conditioning on
accepted images changes the estimand to a selected subpopulation.

The same scene seeds are used across cases, so results are paired and correlated.
This is a small instrument diagnostic, not a Stage 0 pass, human audit, or
validation on generated images. It does not establish that these perturbations
represent real model failures, and it does not test overlap, texture or occlusion.

The checker first applies its original clean-image rules. If those reject an
image, it may retry after scaling a dominant neutral background in the range
225–254 back to white. This is a narrowly calibrated correction for uniformly
darkened white-background scenes; it cannot establish robustness to arbitrary
lighting, background changes, blur, or model-generated images. Ambiguous cases
can still be rejected, and corrupted images cannot all be detected reliably.

If the original and dark-background checks both reject, the checker also tries
two palette-distance thresholds, 115 and 130. It only accepts if both pass the
same color and shape checks and agree on all object labels and exact centers.
These thresholds were explored on synthetic development seeds; their agreement
is a conservative check, not proof that a real or model-generated image was
classified correctly. Remaining ambiguous cases are rejected.

The checker also has a conservative sharpening fallback. It is attempted only
when the original palette validation passes. Two mild `UnsharpMask` settings
with radii `0.5` and `0.6` and percent `100` must return identical object labels
and exact centers;
otherwise the image is rejected. This fallback does not validate generated
images, pass Stage 0, or ensure robustness to blur, lighting, or model output.
