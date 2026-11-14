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

An additional bounded `resume` run was completed from the 8,000-update
checkpoint with `scripts/run-mosaic-pilotunet-resume.ps1`: 12,000 updates,
checkpoints at cumulative 14,000 and 20,000, and no automatic previews during
resume. It restores saved optimizer and random states; the source run did not
enable strict deterministic algorithms, so bitwise CUDA replay is not
established. The one-cell run does not constitute a MOSAIC replication.

The resumed run completed through cumulative 20,000 updates. The 14,000 and
20,000 checkpoint hashes matched the downloaded report; last-100 mean losses
were 0.04799 and 0.04242 respectively. A 14,000-update red preview and paired
20,000-update red/blue previews at seeds 81380–81383 were generated separately
from the saved checkpoints. Informal inspection found small colored regions,
sometimes with no obvious requested color; the 81383 red and blue requests
both produced a green-looking object. Other blue requests did produce blue
regions. These are eight exploratory images, not independent human ratings,
a performance estimate, or proof of conditioning failure. They show the
preview is not ready for a formal checker or MOSAIC gate. The checkpoints and
images are under ignored `outputs/mosaic-pilotunet-resume-81378/` and
`outputs/mosaic-pilotunet-resume-preview-81378/`.

A further bounded resume from the verified 20,000-update checkpoint added
10,000 updates with snapshots at 25,000 and 30,000. The command is recorded in
`scripts/run-mosaic-pilotunet-resume-30k.ps1`. Its purpose is to check whether
conditional generation improves with further training on the same one-cell,
200-image dataset; it is not a Stage 0 replication. It restores saved states,
but strict deterministic GPU replay is not established. It completed, and both downloaded checkpoint hashes matched the report.
Last-100 mean losses were 0.04097 at 25,000 and 0.04053 at 30,000. These
values are not evidence of sample accuracy. Outputs are stored under ignored
`outputs/mosaic-pilotunet-resume-81379/`.
