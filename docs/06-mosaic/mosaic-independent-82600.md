# Second matched model-seed block

The first four 500,000-update models share one training-seed block, so their
image-level variation cannot establish variation between model training runs.
This second exploratory block uses four fresh models on the same 200-image
per-cell dataset. It matches the first block's width 16, batch size 4, 1,000
DDPM timesteps, AdamW settings, velocity objective, EMA and 500,000-update
budget. The initialization and initial training seed is 82600; after the first
1,000 updates the optimizer, model and EMA are restored with fresh RNG seed
82601 because the initial probe lacks RNG state. Subsequent checkpoints save
optimizer and RNG state every 10,000 updates. This is a second model-seed block,
not an independent dataset or a full reproduction of MOSAIC.

The four cells run sequentially under `MuadibMosaicIndependent82600`, with
outputs under `outputs/mosaic-distributed-82600/`. Its namespace is distinct
from the completed 81570 block. A shared workspace supervisor lock and process
check prevent overlapping supervised training on this laptop. Ordinary sleep
suspends execution; after reboot, sign-in is required for automatic restart.
The task retries every five minutes for one year. Failed or incomplete segments
are preserved, and recovery validates checkpoint configuration, code hashes,
CUDA/PyTorch compatibility and recorded report hashes where available.

Before viewing the block's generated images, reserve generation seeds
83000–83007, colors red and blue, guidance 1, EMA and 1,000-step DDPM sampling,
matching the first block's design but not its noise seeds. Generate all four
cells without selecting images. Human review and a validated checker are still
required, as are a defensible exposure design and a comparison of the adapted
contrast with MOSAIC. Two model-seed blocks give very limited information
about training-seed variation and cannot alone establish the full result.

The first block previously required approximately 824 seconds per 10,000
updates on this GPU. Four new 500,000-update runs therefore may take roughly
46 hours plus startup and previews; actual throughput can differ. Keep the
laptop powered and signed in. Interruption can lose work since the last saved
checkpoint; it does not affect the completed first block.

`MuadibMosaicIndependentEval83000` is installed separately and waits for the
training task. It verifies all four 500,000-update checkpoint reports, then
generates the reserved 64 images and prepares blinded packs with frozen checker
predictions on the laptop. Sampling attempts use exclusive directories;
completed cell reports are validated and reused after interruption, while
partial attempts are preserved. It retries at sign-in and every five minutes
and disables itself on successful completion. The resume workflow passed a
mocked end-to-end test on the GPU laptop; actual evaluation cannot be
validated until training reaches its targets. Scheduling the task alone does
not establish evaluation success.

The namespace and checkpoint-recovery tests passed on the GPU laptop. The
one-time installation started `MuadibMosaicIndependent82600`; its log showed a
new low-balanced initial run in the separate 82600 output directory, and CUDA
activity was observed. No checkpoint had been completed at launch verification.
The low-balanced initial run subsequently completed 1,000 updates; the
supervisor verified checkpoint SHA-256
`0c27d4772cfd769157c6c720504d619ef44add61dd7ad0c0a5426fa445dde7a9`
and started the 1,000-to-10,000 continuation. GPU activity was observed.
It subsequently reached a verified 20,000-update low-balanced checkpoint,
SHA-256 `b55919fbb786893e274802c0864006d23f562fc9f6d52aadef53a80a8dcb9214`,
and the training task remained running.
On a later check it had reached a verified 60,000-update low-balanced
checkpoint; the training task and waiting evaluation task were both running.
On the following check, both low-complexity models had reached 500,000
updates and the high-balanced model had a verified 250,000-update checkpoint.
The high-imbalanced cell had not started; training and the waiting evaluation
task were both running.

## Completed block and review pack

On 19 November, the laptop log recorded all four cells at verified 500,000
updates. The automatic evaluation produced the reserved 64 images and four
blinded reviewer archives. The artifacts were copied to the local ignored
`outputs/mosaic-independent-eval-500k-83000/` and
`outputs/mosaic-independent-review-500k-83000/` directories. The copied
protocol, all four report hashes, 64 image and pixel hashes, and all image
entries in the four archives were checked against the frozen provenance.
The provenance SHA-256 is
`9d9c20d5a769a9f379731eb7c3ebc62ae09d9d4be65eced309e6dd45d3b74919`;
the frozen checker predictions SHA-256 is
`8bd0300e3598c4736430f46e4c93e6518ee8b3487942c4a10126280d6f39a7da`.
Hashes verify consistency with the saved artifacts, not independent authorship
or the truth of human labels.

Each reviewer has a separate local folder, `reviewer-1/` through `reviewer-4/`.
Open that folder's `review.html` to view all 64 images and questions, and fill
its `reviews.csv` with `yes`, `no`, or `unsure` for all three questions on
every row. The HTML is a visual reference; answers are entered into the CSV.
Count distinct recognizable objects; use `unsure` if the image is ambiguous.
Each reviewer should work independently and return their own CSV without
discussing answers until all reviews are complete. The four archives contain
the same images, as intended; this does not establish independent reviews.
Keep `private/` and frozen checker predictions away from reviewers.

After the four independently completed files are returned, validate and score
with `exposure.mosaic_matched_eval_score`, passing both frozen hashes above.
The second block shares the original training dataset, so it probes model-seed
sensitivity, not independent dataset sampling. Completion of generation is not
a passed checker, exposure-design, or Stage 0 gate.

After verifying the completed local copies, the two completed scheduled tasks
were disabled on the laptop to prevent repeated five-minute launches. The
checkpoint and evaluation outputs remain in place. The evaluator's last task
result had reported failure despite its log repeatedly reporting packs ready;
artifact integrity was checked separately as described above.

## Returned reviews and descriptive results

The user confirmed that the reviews were independently completed. All four CSVs contain 64 complete
rows and have the same SHA-256
`cfa42d9125121da82df70b8247fe5172329b053572bda5b1534fa7b5ebab6ac8`.
Each of the 256 images in the extracted reviewer folders matched the frozen
provenance; the scorer also verified the pinned provenance and prediction
hashes. The files contain no identifying metadata and cannot independently verify reviewer identities or independence; the independence statement is based on the user’s confirmation. Identical files do not by themselves prove whether reviews were independent.

For requested-color circle presence, counts are:

| Cell | Yes | No | Unresolved |
| --- | ---: | ---: | ---: |
| Low / balanced | 13 | 3 | 0 |
| Low / imbalanced | 16 | 0 | 0 |
| High / balanced | 15 | 1 | 0 |
| High / imbalanced | 14 | 2 | 0 |

There are 58 yes and six no labels. For the secondary expected-object-count
question, yes/no counts in the same cell order are 13/3, 14/2, 10/6 and 10/6.
Object count was not an input condition. The equal-cell descriptive complexity
drop is 0.0; the imbalance drop is -0.0625; their difference is 0.0625. There
are no unresolved target-circle answers in this block, so these finite-image
bounds collapse to those values. They are not confidence intervals.

The first block's complexity drop was 0.0625–0.125, while this block's is
zero. In both blocks the complexity drop was numerically larger than the
imbalance drop, but two small training-seed blocks share one dataset and have
different generated-noise seeds. These observations do not establish a
population effect, a full MOSAIC replication, or a passed gate.

The standard checker answered 8/64 (7 true-positive and 1 true-negative
relative to the submitted labels) and abstained on 56. The experimental
variable-count checker answered 43/64 (41 true-positive and 2 true-negative)
and abstained on 21. All accepted answers agreed with those labels, but four
of six human-negative cases were abstained by the experimental checker. Its
two accepted negative cases are too few to validate a false-accept rate, and
coverage remains selective. The ignored scored artifact is
`outputs/mosaic-independent-review-500k-83000/score.json`, SHA-256
`3a629ab61e261c773060bc67d34209e901872712ced5bb9bcbbd8f47b616dcae`.
