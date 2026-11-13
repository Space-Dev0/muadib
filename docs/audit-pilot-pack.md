# Four-condition pilot audit packs

This workflow prepares a **restricted pilot**, not the full benchmark's Stage
0 checker audit. The source is the 800,000-update model's 300 generated PNGs:
four conditions (`red`, `square`, `red_left_of_square`, `blue_above_circle`),
75 seeds 20000–20074 per condition. The model covers only two complexity
strata, and the same sampling seeds are reused across conditions. The images
are not 300 independent training runs and do not cover all proposed rules or
predicates.

After the image-generation task completes, copy all 300 PNGs into a local
source directory and use the verified local checkpoint copy:

```sh
uv run python -m exposure.audit_pilot_pack \
  --source outputs/audit-pilot-800000-seeds-20000-20074 \
  --checkpoint outputs/archived-checkpoints/velocity-online-v1-800000-79c331f.pt \
  --output outputs/audit-pilot-packs-800000
```

The command requires all expected 64×64 RGB PNGs, refuses to overwrite its
output, and writes four reviewer ZIPs plus a `private/` directory. Give each
reviewer only their own ZIP. The 40 shared images (10 per condition) appear in
all four ZIPs; each of the other 260 images appears in exactly one ZIP. Each
reviewer receives 105 images, 105 whole-rule questions, and approximately 210
separate predicate questions. All four reviewers independently judge the
shared items, while the remaining items have one review each. The protocol
does not itself verify independence or resolve disagreements.

The reviewer ZIPs contain images, human-readable questions, and blank CSVs;
they do not contain predictions, source filenames, generation seeds, or
checkpoint metadata. Keep `private/` from reviewers until independent review
is complete. It contains full and per-reviewer manifests and checker
predictions, image and source hashes, shared IDs, and exact duplicate-pixel
groups. Images are re-encoded from RGB pixels to strip source PNG metadata;
both source-file and reviewer-image hashes are retained privately. A hash of
the supplied checkpoint is recorded, but the packager does
not establish that the images were generated from that checkpoint; retain the
generation script, logs, and source/runtime records. It reports duplicate
images rather than silently counting them as independent.

Return completed rule and predicate CSVs separately for each reviewer. Score
all four numbered reviewer pairs after review using:

```sh
uv run python -m exposure.audit_pilot_score --pack outputs/audit-pilot-packs-800000 \
  --reviewer-1-rule outputs/reviews/r1-rule.csv --reviewer-1-predicate outputs/reviews/r1-predicate.csv \
  --reviewer-2-rule outputs/reviews/r2-rule.csv --reviewer-2-predicate outputs/reviews/r2-predicate.csv \
  --reviewer-3-rule outputs/reviews/r3-rule.csv --reviewer-3-predicate outputs/reviews/r3-predicate.csv \
  --reviewer-4-rule outputs/reviews/r4-rule.csv --reviewer-4-predicate outputs/reviews/r4-predicate.csv \
  --output outputs/reviews/pilot-report.json
```

Each CSV must match that numbered reviewer's exact assigned IDs. The scorer
checks frozen private artifacts, images and archives against their recorded
hashes, reports per-reviewer scores, unique-image coverage, and agreement on
shared rule and predicate questions only. Hashes do not establish which images
reviewers actually saw. It does not infer consensus or pool duplicated reviews
as independent. Discuss an adjudication protocol before resolving disagreements;
do not overwrite independent responses. Rejection is an abstention and must
remain visible alongside conditional error rates. Small or absent per-rule
negative denominators cannot establish a 2% error bound.

This package records standard-checker predictions. A comparison with another
checker requires its own predictions, frozen and kept private before labels
are collected. None of these outputs establishes the full Stage 0 gate.
