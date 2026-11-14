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
longer run's results are recorded separately when it completes.
