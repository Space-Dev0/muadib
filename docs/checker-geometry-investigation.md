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

## Fresh restricted validation pack

A fresh four-condition pilot was generated from the existing 800,000-update
checkpoint using seeds 30000–30009 per condition and
`scripts/generate-audit-validation.ps1`. The frozen candidate is the code in
commit `bc36766`. Forty images were packaged with assignment seed 81219:
eight are shared by all four reviewers, and each reviewer receives 16 images.
Images and assignments are in ignored `outputs/audit-validation-packs-800000/`;
only the four reviewer ZIPs should be distributed. The `private` directory
contains checker predictions and provenance and must not be shared with reviewers. The
packager records image and standard-prediction hashes; it identifies the
supplied checkpoint by hash but does not independently prove image origin.

Before human review, variable-count predictions were saved separately as
`private/variable-count-rules-predictions.json` (SHA-256
`f895ea3443625dc74566715e132f4812eedd70b07d47f348c68eea950152e47f`)
and `private/variable-count-predicates-predictions.json` (SHA-256
`d1f977854561d8cb11b14f4ebdcbc1422f6c5068b85485e9102c18054ed4b5f3`).
This small, fresh validation checks the selected four-rule pilot only. Seeds
are shared across conditions, reviewer independence must be maintained, and
its small size will limit precision. It cannot establish the full benchmark
checker gate. Assemble independent answers and adjudicate disagreements before
comparing either checker's predictions.

## Fresh validation results

The returned four review pairs matched their assignments and all frozen pack
and candidate-prediction hashes. The four reviewers agreed on all eight shared
whole-rule questions and 16 shared predicate questions. The other 32 images
were each reviewed once; the files do not prove the reviewers worked independently.
No adjudication was needed for the shared questions.

| Frozen checker | Whole-rule answered | Abstained | Disagreements among answered |
|---|---:|---:|---:|
| Standard | 19/40 | 21/40 | 0/19 |
| Variable count | 28/40 | 12/40 | 0/28 |

The variable-count checker answered 7/10, 9/10, 6/10 and 6/10 for blue above
circle, square, red, and red left of square, respectively. Of its 12
abstentions, five were human-positive and seven human-negative. For predicate
questions, standard answered 39/80 and candidate 61/80, each with zero
accepted-answer disagreements. Those 80 labels are clustered within 40 images,
and seeds are reused across the four conditions. They are not 80 independent
observations. Both relation singleton questions were positive in all ten
samples for each relation, so these data do not test detection of negative
relation singletons. Review labels are a reference, not error-free ground truth.

This is a new sample for a frozen checker, but only ten seed blocks and four
conditions. Zero observed errors does not establish zero error, and neither
this sample nor the earlier development results justify changing the standard
checker or declaring the full Stage 0 gate passed. Improve validation with
predefined negative and near-tie spatial examples, occlusion and segmentation
cases, a broader rule/complexity sample, and more independently labeled seed
blocks. Preserve the existing rule semantics and report selection separately.

### Relevant research and implications

[Selective classification](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)
frames the key tradeoff as coverage versus error among accepted predictions.
For this project, additionally report label distributions and coverage by rule
and class: an accepted-only success rate estimates a selected population, and
the abstentions cannot be silently discarded. The paper's guarantees do not
automatically apply to our fixed heuristic or this dependent sample.

[GenEval](https://arxiv.org/abs/2310.11513) uses object-focused checks for
position, count, color and related compositional properties and compares them
with human judgments. This supports testing individual failure modes and human
agreement; its reported results do not validate our checker. The controlled
[MOSAIC study](https://arxiv.org/abs/2605.00273) investigates multi-object
attribute, spatial and counting behavior under controlled datasets and model
settings. Its experimental settings provide a useful independent comparison,
but our short toy run is not a replication. Together these suggest targeted,
pre-specified negative spatial examples and class-conditional checker audits,
not an assumption that training longer alone fixes either checker selection or
the exposure design.

As a selection sensitivity illustration, the candidate accepted 21 positive
and seven negative whole-rule answers and abstained on 12. If its accepted
answers were assumed correct but the abstentions had unknown labels, the
positive fraction for these exact 40 images could be anywhere from 21/40
(52.5%) to 33/40 (82.5%). The complete human labels give 26/40 (65%) for
this sample. The interval is a finite-sample worst-case bound under the
accepted-answer assumption, not a confidence interval or proof of checker
correctness. [Horowitz and Manski](https://www.sciencedirect.com/science/article/pii/S030440760500062X)
discuss bounds when missing data prevent point identification; they do not
specifically validate this checker.

The machine-readable scoring report is stored locally at ignored
`outputs/audit-validation-results-800000.json`. To reproduce it, pass the
pack and each review pair to `python -m exposure.audit_validation_score` with
`--reviewer-N-rule`, `--reviewer-N-predicate` (N = 1–4), and a new `--output`
path. The scorer checks the frozen hashes and assignments and excludes any
reviewer-disputed item instead of inventing a label.
