# Exploratory checker geometry investigation

The eight seed-201/202 800,000-update review images were evaluated with the
frozen checker and reviewed by four people. All reviewers gave the same
whole-rule answers. Five images were rejected. Four contained an 11×11
component with a fill ratio near 0.55, visually resembling a triangle; the
default checker requires width 12. The remaining rejected image contained a
12×12 component with fill ratio 0.924, between its circle and square ranges.
These are image component descriptions, not independently verified object
labels. Images in different rule conditions share seeds; they must not be
assumed independent.

An in-memory trial allowed width 11 only for an 11-high component with fill
ratio 0.49–0.60, leaving the 12-wide rule unchanged. On the eight images it
accepted seven requested-rule evaluations, all matching the unanimous human
answers; one image still abstained. These same images informed the trial, so
they are **development data**, not a holdout or a gate result. On the earlier
16-image calibration, the trial accepted nine evaluations that agreed with
one reviewer's original answers and abstained on seven; reviewers disagreed on
two of those questions, so that is not an adjudicated accuracy estimate. A
separate clean-render check of seeds 44000–44199 produced zero detector
failures and zero rule errors among the default audit rules. Clean renders do
not establish correctness on generated images or unsupported geometry.

An opt-in `detect_narrow_experimental(image)` now implements the narrow
component trial in the checker without the standard detector's fallbacks.
It assumes a one-pixel center uncertainty radius for each narrow component and
abstains if any pair involving it has x or y distance at most the sum of the
assumed radii. The standard `detect` and all existing audit workflows remain
unchanged. This radius is a modeling assumption, not a verified bound for
generated images. On the same eight development questions the experimental
detector accepted seven and matched their unanimous whole-rule answers; no
independent validation is implied.

A further synthetic check at previously unused seeds 45000–45199 applied
brightness 1.1 and blur radius 0.5. The experimental detector accepted 157/200
and 145/200 scenes respectively; none of its accepted scenes disagreed with
renderer annotations on the nine default audit rules. These are synthetic
scenes with a limited rule panel, not generated-image or position validation.
The independent OpenCode review also observed half-pixel center differences on
perturbed triangles, reinforcing that the uncertainty radius is an assumption.

After freezing the experimental detector, eight additional images were
generated from the 800,000-update checkpoint using seeds 301 and 302 across
the four pilot conditions. The standard and experimental predictions and
checker/image hashes were saved privately before review in ignored
`outputs/geometry-holdout-800000/`. The separate blinded reviewer archive is
`outputs/geometry-review-pack-800000.zip`. Reused seeds create dependence
across conditions. This small follow-up cannot replace the formal audit or
certify the assumed uncertainty bound.

An 11-pixel bounding box gives a
half-pixel center under the current estimator. A boundary or occlusion can move
that center, potentially flipping a spatial relation. Before considering a
change to the standard audit, validate the center uncertainty bound; test tied and
near-tied relations, overlapping and clipped shapes, unsupported geometry,
mixed colors, and clean scenes with known annotations. Then use a fresh,
independently human-labeled image set selected after the candidate is frozen,
with both requested and other relevant rule questions. Report checker
rejections and errors separately by rule and class. The current model's four
rules and two complexity strata do not cover the proposed benchmark.
