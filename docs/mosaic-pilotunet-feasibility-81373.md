# PilotUNet one-cell feasibility probe

OpenCode on the GPU laptop implemented `exposure.mosaic_pilotunet_probe` and
six focused tests. The probe uses the existing PilotUNet, velocity target and
DDPM velocity-to-noise sampler on the 200-image low/balanced synthetic cell.
It conditions on target color; in this restricted dataset red and blue targets
are always circles. It is not a four-cell MOSAIC experiment or a Stage 0 gate.

The first 8,000-update OpenCode background process ended when its OpenCode
session exited. Its incomplete output `mosaic-pilotunet-probe-81370` was
preserved. A separate SSH-attached run at `mosaic-pilotunet-probe-81373`
finished 2,000 updates on the RTX 3050, width 16, batch 4, seed 81370 and
1000 diffusion timesteps. It saved checkpoints at 1,000 and 2,000 updates and
paired red/blue previews using sample seed 81380. Both checkpoint hashes were
verified after copying the outputs to ignored local
`outputs/mosaic-pilotunet-probe-81373/`. The last-100 mean training loss was
0.0952 at 2,000 updates; the 2,000-update paired previews appear as noise.
Two images and this short budget cannot establish that the architecture will
fail at longer training. Palette and component metrics in the report are
exploratory diagnostics, not validated shape accuracy.

The checkpoints store model, EMA and optimizer state but do not store RNG
state and declare themselves non-resumable. Any subsequent run starting from
these weights must use a new RNG stream and cannot claim exact continuation.
The architecture, schedule and training configuration differ from the earlier
multiscale probe, so the previews are not an isolated architecture comparison.

Local focused validation: `uv run --extra train python -m unittest tests.test_mosaic_pilotunet_probe -q` (the GPU laptop's CUDA smoke test and the
local CPU tests both passed). The research hypotheses are documented in
[diffusion failure research](diffusion-failure-research.md).
