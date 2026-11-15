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
eight; no independent human judgment has yet validated these predictions.
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

Training remains stopped. This one model is conditioned on red/blue target
color in low/balanced two-object scenes. It cannot by itself establish the
proposal's broader complexity-stratified checker audit, full design
covariates, or four-cell complexity-versus-imbalance replication. Do not claim
Stage 0 has passed from these previews.
