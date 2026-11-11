# 310,000-update human calibration

Four submitted review sets were compared against the exact 16-image preview
and its checker predictions. All four whole-rule files contain 16 valid
judgments and all four predicate files contain 32. The reviewed image hashes
and manifests match the local prediction source. Reviewers' independence
cannot be established from the files alone. Individual submissions and
machine-readable reports are kept in ignored `outputs/`, not published here.

Whole-rule judgments are unanimous for 14/16 images; pairwise agreement is
87.5% to 100%. The two remaining images are `preview-011` and `preview-012`,
both asking whether a red object is left of a distinct square. Both are split
2 yes / 2 no. Predicate judgments are unanimous for 31/32 questions; the one
remaining question is `predicate-000032`, the independently rebound `above`
question for `preview-016` (3 yes / 1 no). These were subsequently
adjudicated as described below; the independent answers have not been changed.

## Adjudication received (2026-11-11)

The team adjudicated all three disputed questions as **yes**, with written
reasons in `outputs/calibration-adjudication-310000/adjudication.csv`. Original
individual answers remain unchanged. A separate sensitivity score combines
unanimous individual answers with these supplied group decisions; it is stored
in `outputs/audit-results-310000-adjudicated/`. This treatment does not make
the adjudicated answers independent judgments.

For whole rules, the checker accepted 5/16 and all five agree with the
adjudicated human answers. It rejected six human-yes and five human-no images.
There are **zero accepted human-no images**, so the accepted-only false-accept
rate remains undefined. If rejected outputs were hypothetically completed as
binary answers, the reviewed-sample false-reject range would be 0–6/11 and the
false-accept range 0–5/5. These are identification bounds, not measured error
rates or confidence intervals. Predicate questions have 11/32 coverage, with
17 rejected human-yes and four rejected human-no questions. They share images
and are not independent observations. Stage 0 remains not passed.

The checker rejected 11/16 whole-rule images outright. Of the five images it
accepted, all predictions were yes. Two reviewers agreed with all five; the
other two disagreed with one accepted prediction (`preview-012`). Therefore
the conditional false-accept rate is undefined for two reviewers (zero
accepted human-negative images), and 1/1 for each of the other two. Neither
figure estimates a reliable population error rate. Among the 11 checker
rejections, reviewers judged five or six images yes, depending on reviewer;
rejection is an important separate coverage outcome, not a validated false
or true classification.

The checker produced binary answers for 5/16 images (31.25%). Treating each
abstention as an unknown binary answer gives reviewed-sample false-reject
completion ranges of 0–54.5% for two reviewers and 0–55.6% for the other two;
the corresponding false-accept completion ranges are 0–100% and 14.3–100%.
These are worst/best case assignments of unknown outcomes, not observed error
rates or confidence intervals, and they depend on each reviewer's labels.
The ranges show why accepted-only errors do not establish the checker gate.

For the three disputed questions, a review-only discussion pack is available
locally at `outputs/calibration-adjudication-310000.zip`. It includes an empty
decision sheet, no checker predictions, and instructions to retain the original
independent reviews. The completed decision sheet is separate from that
original ZIP.

Predicate scoring repeats questions on the same images; its 32 questions are
not 32 independent images. One reviewer disagreed with the checker on one of
11 accepted predicate questions; three agreed on all 11. Raw agreement does
not establish accuracy, a 2% error bound, or independent review. The sample
is a convenience calibration from four rules and two complexity strata; it is
not the proposal's 300-image audit. Stage 0 remains not passed.

Individual reports are in `outputs/audit-results-310000/`; original reviews
are in `outputs/preview-review-pack-310000/` and predictions are separate in
`outputs/preview-310000-20261110/`. Do not expose predictions to reviewers
before their independent review is complete.
