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

The four returned reviewer sets agreed on all eight questions. Frozen standard
predictions accepted 1/8; frozen narrow-detector predictions accepted 2/8.
All accepted answers agreed with the reviewers; five of the six experimental
rejections were uncertain spatial order, and one was unsupported geometry.
The independent reviews and scores are in ignored
`outputs/audit-results-geometry-800000/`. Reviewers' independence cannot be
verified from their files. These counts do not establish population accuracy.

Two of the uncertain-order rejections asked only about attributes. A separate
experimental API, `evaluate_narrow_attributes_experimental(image, predicates)`,
evaluates attribute-only rules without testing spatial order and refuses any
relation predicate. On these two now-development examples it returned the
human answer for each. Object shape/color uncertainty remains; this observation
is not independent validation. Neither experimental function is used by the
standard audit.

An additional experimental API, `evaluate_narrow_rule_experimental`, evaluates
attribute and relation conjunctions using three-valued interval comparisons
and distinct existential bindings. It returns `None` if no binding is
definitely true but an uncertain one remains possible; incompatible predicates
and detector failures are not silently labeled. A ±1-pixel radius for narrow
components is assumed, not empirically guaranteed, and other detected centers
are treated as exact. The conservative interval logic does not model correlations
between multiple uncertain relations and may abstain unnecessarily. On the
same eight questions that motivated this change, it answered seven, all agreeing
with the reviewers, and abstained on one; this is development data, not a
holdout or evidence of research-grade accuracy. Neither standard detection nor
the audit workflow invokes this experimental API.

On clean-render scenes at seeds 46000–46099 with brightness 1.1 or blur radius
0.5, the attribute-only experimental evaluator accepted all 100 scenes in each
condition and made zero errors across the seven default singleton attribute
rules. These synthetic cases do not validate generated images or relations.

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

## Variable component count: development only

`evaluate_variable_count_experimental` also examines two or four separated
components. For those counts it returns a positive only when the detected
objects contain a definite witness; negative conclusions abstain because an
object could be missing or hidden. Three-component behavior uses the existing
experimental rule evaluator. Segmentation, object identity and center
uncertainty are assumptions, so even a positive answer is not guaranteed.
The standard checker and audit remain unchanged.

On the same 300 development images and assembled, partly adjudicated labels,
this candidate answered 226 (186 positive and 40 negative), abstained on 74,
and its answers agreed with the assembled labels. Of the 74 abstentions, 37
had positive and 37 negative human labels. This is an adaptive development
comparison, not independent accuracy validation; the results do not establish
performance on new images, other rules or the full benchmark. A paired
seed-block bootstrap estimated the coverage improvement over the standard
checker as 43.7 percentage points (percentile interval 36.0–51.0), conditional
on exchangeable blocks and without accounting for label or selection uncertainty.
It is not an accuracy interval. Fresh blinded human labels and a frozen candidate
are needed before changing the checker used for research.
