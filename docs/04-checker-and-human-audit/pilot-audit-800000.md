# Human-reviewed four-condition pilot audit

Four reviewers returned complete whole-rule and predicate-question CSVs for
the restricted 800,000-update pilot. The saved reviewer ZIP hashes match the
original four assignments despite the files having been renamed. Private
manifest, prediction, image and archive hashes passed the scorer's checks;
hashes do not establish what each person actually saw or whether reviewers
worked independently. The two copies of one reviewer's files contain identical
judgments. Original files remain unchanged.

The sample contains 300 distinct image IDs, 75 per conditioning rule and two
complexity strata. Forty images are shared by all four reviewers; the other
260 were assigned once. The scorer found no exact duplicate pixel groups.
Seeds were reused across conditions, so the 300 images are not independent
training runs and may be correlated. This limited pilot does not cover the
full benchmark and is not sufficient to pass the Stage 0 checker gate.

On the 40 shared images, reviewers were unanimous on 37 whole-rule questions;
pairwise agreement ranged from 95% to 97.5%. The three disagreements are
`pilot-0009`, `pilot-0022`, and `pilot-0032`. All 80 shared singleton-predicate
questions were unanimous; questions sharing images are not independent images.

The standard checker made a binary whole-rule prediction for 95 of 300 images
(31.7%) and abstained on 205 (68.3%). For the 95 accepted images, human
judgments were unanimous wherever an image had multiple reviewers. Counting
each image once yields 73 true positives, 19 true negatives, one false accept,
and two false rejects relative to the submitted judgments. The three accepted
disagreements occur in `red_left_of_square`; each was reviewed by only one
person, so the human labels should be checked before treating them as final.
Accepted-only false-accept and false-reject fractions are 1/20 and 2/75,
respectively. Their denominators are small and selection excludes abstentions;
these are neither overall error rates nor evidence for the proposal's 2%
target. All three reviewer-disagreement cases were abstained by the checker.

Coverage by conditioning rule was: red 18/75, square 24/75,
red-left-of-square 28/75, and blue-above-circle 25/75. These are descriptive,
not adjusted for seed pairing or other sources of dependence. Separate
per-reviewer and per-predicate reports, including rejection counts, are in
ignored `outputs/audit-pilot-results-800000/score.json`. Shared observations
must not be summed as if independent.

A local discussion pack, `outputs/audit-pilot-followup-800000.zip`, contains
the three reviewer-disagreement cases and three checker-versus-reviewer cases.
It does not expose checker predictions. Because cases were selected after
inspection, follow-up decisions are adjudication, not independent validation;
preserve the original reviews and record any decisions separately.

## Submitted follow-up and development comparison

The submitted follow-up records yes for `pilot-0009`, a qualified “yes but
very very slightly” for `pilot-0022`, no for `pilot-0032`, no for `pilot-0068`
and `pilot-0168`, and yes for `pilot-0191`. The qualified answer was interpreted
as yes for this separate sensitivity calculation; the original wording and
individual reviews are preserved. Whether the decisions represent a group
consensus cannot be determined from the file alone. Cases were selected after
inspection and must not be treated as independent validation.

Using those follow-up answers, unanimous shared answers, and original answers
for other images, the standard checker has 74 accepted human-yes and 21
accepted human-no matches, with no accepted disagreements; it still abstains on
205/300, including 148 human-yes and 57 human-no labels. These are descriptive
post-selection results. The 0/21 accepted-only false-accept and 0/74
accepted-only false-reject fractions do not measure errors on rejected images,
prove a 2% target, or remove the limited benchmark coverage.

An exploratory rule-aware narrow-geometry checker, applied after labels were
available, returned a decision on 166/300 and abstained on 134. It agreed on
165 accepted images and disagreed on one (`pilot-0279`, a blue-above-circle
question reviewed once). This is development analysis of previously reviewed
data, not a blinded validation or a replacement for the standard checker. A
new one-case discussion pack is in
`outputs/audit-pilot-experimental-followup.zip`; no checker prediction is
shown. The per-setting comparison, rejection reasons and label provenance are
in ignored `outputs/audit-pilot-results-800000/`.

The submitted one-case follow-up for `pilot-0279` is **yes**; its image hash
matches the original reviewed image. The original single-reviewer “no” is
preserved. The experimental checker had answered yes, while the standard
checker had abstained. Incorporating this selected follow-up removes that
particular checker-versus-label disagreement, but the case was selected after
seeing outcomes, and its change is not independent validation. The standard
checker still abstains on 205/300 and the experimental checker on 134/300.

`exposure.audit_followup` reproduces the standard-checker sensitivity score
from the full private rule manifest, frozen predictions, four original review
CSVs, and the submitted `sample_id,decision,notes` follow-up CSV. It identifies
cases selected by either reviewer disagreement or checker/reviewer disagreement
and refuses missing, duplicate, unexpected, or unresolved answers. A qualified
`yes but ...` or `no but ...` is interpreted as yes or no while the original
CSV retains its exact wording. It does not independently verify image hashes;
run the pilot scorer's integrity checks first. The separately written
`adjudicated-reproducible.json` matches the initial sensitivity calculation.
