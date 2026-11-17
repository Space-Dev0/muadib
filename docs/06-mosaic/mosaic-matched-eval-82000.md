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
