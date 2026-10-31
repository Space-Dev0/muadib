# Prototype image checker

`exposure.checker.detect` uses image pixels only. It segments palette-colored
components, rejects several unsupported or ambiguous cases, classifies simple
shape geometry, and reconstructs centers for the current renderer. Its checks
are calibrated to 64×64 clean RGB scenes with three separated objects.

Run the clean-render instrument audit:

```sh
uv run python -m exposure.checker --samples 300 --seed 0 --output outputs/checker-audit.json
```

The audit compares detector-based rule evaluations with exact renderer
annotations. Per rule, `positive` and `negative` are the numbers of accepted
scenes with and without the rule; `false_rejects` and `false_accepts` count
errors within those denominators. `detector_failures` are reported separately
and excluded from rule denominators, so low errors among accepted scenes must
not be interpreted without the failure count. Per rule,
`detector_failure_positive` and `detector_failure_negative` classify failed
images by ground truth; failures are not counted as false accepts or rejects.
These counts reveal whether detector coverage differs between positive and
negative scenes. This audit uses a small fixed
selection of unary and relational rules, not all proposed templates.

The detector has **not** been validated on diffusion outputs, overlapping or
touching objects, texture, or realistic perturbations. It cannot guarantee that
all malformed images are rejected. Clean-render thresholds are prototype
choices and the test seeds overlap the development of those choices. These
counts do not constitute an independent checker certification or a Stage 0
pass. Human review of at least 300 generated images and the other checks in the
proposal remain outstanding.
