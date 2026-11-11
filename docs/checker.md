# Prototype image checker

`exposure.checker.detect` uses image pixels only. It segments palette-colored
components, rejects several unsupported or ambiguous cases, classifies simple
shape geometry, and reconstructs centers for the current renderer. Its checks
are calibrated to 64×64 clean RGB scenes with three separated objects.

Detection uses the original palette pass first. Conservative fallbacks include a
narrow dark-background correction, two palette-distance thresholds, and a mild
sharpening retry. The sharpening fallback is attempted only when the original
palette validation passes. It accepts only if two `UnsharpMask` settings with
radii `0.5` and `0.6` and percent `100` return identical object labels and exact centers. This
fallback does not validate generated images, pass Stage 0, or ensure robustness.
Fresh coverage numbers for synthetic seeds 41000–41199 are in
[checker perturbations](checker-perturb.md); rejected scenes are excluded, so
they do not establish overall validation.

An exploratory API option, `detect(image, palette_tolerance=45)`, raises the
default palette-blend residual limit from 40 to 45. On three 800,000-update
preview images rejected for color, each rejection was triggered by one
saturated object pixel with a residual between 40.21 and 42.21. The option
allows two through the remaining checks; the third still fails geometry. On
the earlier adjudicated 16-image preview, acceptance remained 5/16. A
200-scene clean-render check at seeds 43000–43199 had no detector failures.
These are exploratory checks, not an independent generated-image validation;
the standard `detect(image)` and audit workflow retain the limit of 40.
Relaxing the limit changes which colors can be accepted and must be checked
against fresh human-labeled images before use in a research audit.

Four reviewers subsequently labeled all four 800,000-update seed-101 preview
images, unanimously. The standard setting accepted one human-positive image,
rejecting two human-positive and one human-negative image. The experimental
setting accepted all three human-positive images, rejecting the one
human-negative image on geometry. No accepted human-negative image is present,
so accepted-only false-accept rates are undefined for both settings. These
four images include the examples that motivated the tolerance change; they
are development data, not an independent test, and do not validate a gate.
Individual scores and predictions are in ignored `outputs/audit-results-800000/`.

A separate eight-image follow-up was generated from the same checkpoint at
previously unused seeds 201–202 for each of the four pilot conditions. Both
checker settings were run and their predictions, checker code hash, and image
hashes saved privately before requesting labels. The review-only archive is
`outputs/holdout-review-pack-800000.zip`; it excludes predictions. This small
follow-up can test the change on new samples but cannot establish the 2% gate
or replace the 300-image complexity-stratified audit.

Four reviewers returned complete follow-up reviews and agreed on all eight
questions; independence cannot be verified from the files alone. Image hashes
and the checker source hash matched the predictions saved before review. Both
limits produced identical results: three accepted answers agreed with the
reviewers (two human-positive, one human-negative) and five images were
rejected (three human-positive, two human-negative). Four rejections were
reported as unsupported size or occlusion, and one as ambiguous shape. Thus
the wider palette limit produced no coverage gain on this set. This is a small
follow-up, not evidence of population accuracy or a reason to relax geometry
thresholds without independent validation. Reports are in ignored
`outputs/audit-results-800000-holdout/`.

Run the clean-render instrument audit:

```sh
uv run python -m exposure.checker --samples 300 --seed 0 --output outputs/checker-audit.json
```

The audit compares detector-based rule evaluations with exact renderer
annotations. The report includes `rejection_rate`, `coverage`, and per-rule
`conditional_sensitivity`, `conditional_specificity`,
`unconditional_sensitivity`, `unconditional_specificity`,
`positive_coverage`, and `negative_coverage`. Conditional rates use accepted
images only. Unconditional sensitivity and specificity divide correct accepted
classifications by all true positives and negatives respectively, including
detector failures in the denominator. These coverage-adjusted rates do not
assign a predicted class to failures.
Per rule, `positive` and `negative` are the numbers of accepted
scenes with and without the rule; `false_rejects` and `false_accepts` count
errors within those denominators. `detector_failures` are reported separately
and excluded from rule denominators, so low errors among accepted scenes must
not be interpreted without the failure count. Per rule,
`detector_failure_positive` and `detector_failure_negative` classify failed
images by ground truth; failures are not counted as false accepts or rejects.
These counts reveal whether detector coverage differs between positive and
negative scenes. This audit uses a small fixed
selection of unary and relational rules, not all proposed templates.

Pass `--rules examples/exposure_rules.json` to audit the named predicates in
another rule-set file, using the same JSON mapping format as exposure
measurement. The report then includes `rule_definitions` alongside exact counts
for each named rule. A zero positive or negative denominator means that error
rate is unavailable; it does not establish correctness. Without `--rules`, the
default fixed rule selection is used.

The detector has **not** been validated on diffusion outputs, overlapping or
touching objects, texture, or realistic perturbations. It cannot guarantee that
all malformed images are rejected. Clean-render thresholds are prototype
choices and the test seeds overlap the development of those choices. These
counts do not constitute an independent checker certification or a Stage 0
pass. Human review of at least 300 generated images and the other checks in the
proposal remain outstanding.
