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
