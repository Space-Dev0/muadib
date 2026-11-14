# PilotUNet warm-start and resume probe

OpenCode on the GPU laptop implemented `exposure.mosaic_pilotunet_warmstart`
and its tests. A warm start from the prior 2,000-update checkpoint restores the
model, EMA and AdamW optimizer but creates new random streams, because the old
checkpoint did not contain RNG state. It is not an exact continuation. New
checkpoints include optimizer, Python, NumPy and Torch random states; a CPU
unit test compares resumed and uninterrupted training. Exact GPU replay has not
been verified and can depend on deterministic backend settings and runtime.

A bounded one-cell run starts from the 2,000-update checkpoint, adds 6,000
updates with fresh seed 81377, and saves at cumulative 5,000 and 8,000 updates.
The dataset, network width, batch size, optimizer settings and sampler remain
those of the source checkpoint. Paired 1,000-step previews use seed 81380.
This is a learnability experiment, not a four-cell MOSAIC replication or
Stage 0 gate. It runs over an attached SSH session because OpenCode's earlier
background process was terminated when that session ended. A network
interruption could stop it; preserve its outputs and use only complete
checkpoints.

Local focused tests: `uv run --extra train python -m unittest tests.test_mosaic_pilotunet_warmstart -q`.
The initial CUDA smoke test and CPU replay test ran on the GPU laptop. The
completed longer run is described below.

The CUDA warm start completed 6,000 added updates (cumulative 8,000). Local
copies of both checkpoint hashes matched the report. Last-100 mean losses
were 0.06079 at 5,000 and 0.05617 at 8,000. The fixed-seed 8,000 preview
shows a few colored blobs against mostly background, still not reliably
recognizable objects. This is an informal inspection of two images, not a
quality metric or evidence the model has converged. Checkpoints and previews
are stored in ignored `outputs/mosaic-pilotunet-warmstart-81377/`.

An additional bounded `resume` run was started from the 8,000-update
checkpoint with `scripts/run-mosaic-pilotunet-resume.ps1`: 12,000 updates,
checkpoints at cumulative 14,000 and 20,000, and no automatic previews during
resume. It restores saved optimizer and random states; the source run did not
enable strict deterministic algorithms, so bitwise CUDA replay is not
established. The one-cell run does not constitute a MOSAIC replication.
