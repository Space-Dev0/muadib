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
