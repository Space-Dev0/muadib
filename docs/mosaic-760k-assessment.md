# Stopped 760,000-update model: exploratory assessment

On the user's instruction, the continuous training stop script was run. The
continuous task was disabled, the predecessor task was Ready, no Python
training process remained, and GPU utilization and allocated memory were zero
before inference. The supervisor state recorded 760,000 updates; an inventory
of later segment folders found no newer checkpoint. The checkpoint SHA-256
`84dac304a3a2b67ca8342f331b05adea940da746293dd7a3002bb902c43d91d4`
matched both the recorded state and the local copy at
`outputs/model-eval-760k/warmstart-760000.pt`. A partially completed later
segment, if present, was preserved. This is the single-cell PilotUNet model,
not the earlier separate four-rule 800,000-update online-velocity model.

`exposure.mosaic_guidance_probe` generated eight 760k images: red and blue,
seeds 81380 and 81381, guidance 1 and 2, using EMA weights and the 1,000-step
DDPM sampler. These are paired with prior 30k and 150k examples. All eight
image hashes and the checkpoint hash match the report. Informal inspection
found separated colored objects in the displayed examples. The experimental
variable-count checker returned true for the requested-color circle on all
eight. The returned human review files also mark the target circle present on
all eight; see the review results below.
These selected, paired images are not a population accuracy estimate.

The earlier 150k comparison contains eight matched and four additional-seed
images. All image hashes matched their reports. Across those 12 images, the
experimental checker returned three true decisions and abstained on nine;
checker abstention is not model failure and accepted decisions are not ground
truth. None of the 150k or 760k comparison images was an exact pixel duplicate
of any of the 800 training images in the four-cell data manifest. Exact pixel
matching does not rule out near-memorization or other forms of overfitting.
The model's own training cell contains only 200 fixed images and has been
repeatedly sampled, so evaluate generalization independently.

A blinded eight-image readiness pack for the 760k matched previews is at
`outputs/mosaic-guidance-review-760k-81411/reviewer-1.zip` through
`reviewer-4.zip`. Reviewers independently answer whether the requested-color
circle and exactly two objects are visible, using yes/no/unsure. Guidance,
seed and checker decisions are private. Archive and image hashes were verified.
The pack was selected after looking at the samples; one model, two paired seeds
and two guidance settings cannot establish the checker-error gate or MOSAIC
replication. It is a readiness check only.

## Returned reviews

Five completed exports (`reviews.csv` and `reviews_1.csv` through
`reviews_4.csv`) were found in the pack directory. Each contains exactly the
eight expected IDs, with `yes` for both questions on every image. All five
files have the same SHA-256
`dc5104a4a1add23035a0690a31360fe8a20de16d0c415ed6a99cdd012b9b8e14`;
the filled `reviewer-1/reviews.csv` is identical as well. Identical answers
are consistent with agreement, but the exports contain no reviewer identifier
and the reviewer archives are identical, so the number of independent
reviewers or duplicate exports cannot be established from these files alone.
No disagreements or unsure answers are recorded. The results match the
previously recorded experimental checker decisions for target-circle presence;
they do not independently validate the checker's object-count output.

The image file hashes in the private manifest match the eight images in each
of the private, reviewer-1 and reviewer-2 directories. The source report and
checkpoint hashes match the private provenance. The review CSVs remain in the
ignored output directory. This post-hoc, eight-image convenience sample does
not estimate population error, measure false accepts and false rejects, or
meet the complexity-stratified 300-image checker audit.

Training remains stopped. This one model is conditioned on red/blue target
color in low/balanced two-object scenes. It cannot by itself establish the
proposal's broader complexity-stratified checker audit, full design
covariates, or four-cell complexity-versus-imbalance replication. Do not claim
Stage 0 has passed from these previews.
