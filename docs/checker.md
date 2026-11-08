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
