# Bounded resumable multiscale probe

This one-cell feasibility run starts from the `+30000` checkpoint in
[multiscale warm start](mosaic-warmstart-probe.md), which itself began from a
20,000-update model. The parent does not contain optimizer or random state, so
the start of this run uses fresh AdamW and RNG state. It is not an exact
continuation from the original 20k checkpoint. The run performs 30,000 further
updates on the low-balanced cell and saves checkpoints and matched 50-step
previews every 10,000 updates. It does not establish a MOSAIC contrast or a
Stage 0 pass.

New checkpoints record model, optimizer, Python and Torch RNG states, parent,
dataset and source hashes, seed and runtime. A two-step CPU test verifies that
a resumed run matches an uninterrupted run under that test's conditions.
Exact GPU replay is not established: backend determinism and runtime details
can affect results. The saved model file is written through a temporary file
and atomically renamed. Preview files follow checkpoint creation, so an
interruption can leave a valid checkpoint with missing previews.

The laptop's one-off scheduled task was interrupted twice with Windows return
code 3221225786 and an empty log before output was created. The same script was
then run over SSH, where GPU utilization confirmed training started. This is a
bounded task, not an automatically restarting continuous trainer. The run and
artifacts live under `outputs/mosaic-resumable-probe-81360` on the laptop.

If interrupted after a checkpoint, from the staged source folder run the module
with the same dataset, parent, seed, device, source and Torch version, add
`--resume outputs/mosaic-resumable-probe-81360/resumable-plus-10000.pt` (or the
latest complete checkpoint), and use a `--steps` target greater than that
checkpoint's step. Existing checkpoints and previews are never overwritten.

## Result

The SSH-attached run completed all 30,000 updates. All three downloaded
checkpoint hashes matched their corresponding reports. Mean losses over the
last 500 updates were 0.01288, 0.01187 and 0.01016 at +10k, +20k and +30k.
The locally stored red seed-81330 previews still have merged/distorted regions;
this is an informal inspection of one matched seed, not an accuracy estimate.
Loss and additional updates do not establish recognizable generation. The
three checkpoints and previews are backed up under ignored
`outputs/mosaic-resumable-probe-81360/`.
