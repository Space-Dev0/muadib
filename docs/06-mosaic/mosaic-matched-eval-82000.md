# Fixed 500k matched-cell evaluation

Before viewing these samples, fix the four locally verified 500,000-update
PilotUNet checkpoints, target colors red and blue, guidance 1, EMA weights,
1,000-step DDPM sampling, and seeds 82000–82007. Generate every combination:
16 images per cell and 64 images total. Noise seeds are paired across colors
and cells; they are not independent model-training replicates. The training
conditions use one matched training-seed block and 200 fixed images per cell.

Primary descriptive success is the human-reviewed presence of a requested-color
circle, averaged equally over red and blue within each cell. Secondary labels
record exactly two and exactly three visible distinct objects; low-complexity
training scenes contain two and high-complexity training scenes contain three.
The generation condition supplies target color only, so object count is not
an additional prompt and is not part of the primary outcome. Reviewers should
use yes/no/unsure independently and disagreements must remain visible.

Calculate descriptive run-level complexity and imbalance drops using
`exposure.mosaic_contrast` only after human review. Retain per-cell and per-color
counts, unsure responses, checker abstentions and disagreements. One model per
cell cannot establish training-seed variability, and this small 2D adaptation
cannot establish the full MOSAIC result or the Stage 0 checker gate. Do not
interpret training loss, unchecked generated images, or checker abstentions as
human success or failure. Further independent training-seed blocks and a
validated evaluation method are required for a robust direction comparison.

`exposure.mosaic_matched_eval` validates each checkpoint against its original
report, expected hash, cell, training configuration and source fingerprints;
then writes `protocol.json` before generating images. It writes exclusive
output folders and a completion marker only after all reports are complete.
Interruption can leave partial data that must be preserved. The Windows task
is one-shot and does not automatically resume after interruption.

All 64 images and reports were generated and copied to
`outputs/mosaic-matched-eval-500k-82000/`. The blinded reviewer packs are in
`outputs/mosaic-matched-review-500k-82000/reviewer-1.zip` through
`reviewer-4.zip`; the team need not review them until available. All image,
pixel, report and checkpoint hashes were checked while packaging. Frozen
checker predictions are private; their SHA-256 is
`fba809ab332aa30ac7664c745fb20c0b68b63ac984c0310f780a3e76b43114d0`.
The private provenance SHA-256 is
`08e9dbca49143e15decca05205b2f74f8146b292e08c9a0b82df5188fc53907d`.
Hashes document the artifacts, but do not independently attest that a
reviewer saw an image or worked independently.

After all reviews are returned, use four distinct review CSVs:

```sh
uv run python -m exposure.mosaic_matched_eval_score \
  --pack outputs/mosaic-matched-review-500k-82000 \
  --reviews reviewer1.csv reviewer2.csv reviewer3.csv reviewer4.csv \
  --expected-predictions-sha256 fba809ab332aa30ac7664c745fb20c0b68b63ac984c0310f780a3e76b43114d0 \
  --expected-provenance-sha256 08e9dbca49143e15decca05205b2f74f8146b292e08c9a0b82df5188fc53907d \
  --output outputs/mosaic-matched-review-500k-82000/score.json
```

The scorer retains individual votes and counts disagreements and unsure
answers as unresolved. It reports worst-case bounds for these 64 images, not
confidence intervals, and compares frozen checker answers only against
unanimous yes/no decisions. No automatic adjudication is performed.
