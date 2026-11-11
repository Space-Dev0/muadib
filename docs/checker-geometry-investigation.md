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

This trial is not used by the checker. An 11-pixel bounding box gives a
half-pixel center under the current estimator. A boundary or occlusion can move
that center, potentially flipping a spatial relation. Before considering a
change, define center uncertainty and abstention for near-ties; test tied and
near-tied relations, overlapping and clipped shapes, unsupported geometry,
mixed colors, and clean scenes with known annotations. Then use a fresh,
independently human-labeled image set selected after the candidate is frozen,
with both requested and other relevant rule questions. Report checker
rejections and errors separately by rule and class. The current model's four
rules and two complexity strata do not cover the proposed benchmark.
