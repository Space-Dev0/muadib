# Matched four-cell PilotUNet experiment

The user authorized a new bounded four-condition experiment after the previous
760k continuous run was stopped. The old STOP marker, supervisor, checkpoint and
outputs are untouched. The new task is `MuadibMosaicMatched81570` on the Windows
GPU laptop, and writes a transcript to `%USERPROFILE%\muadib\mosaic-matched-81570.log`.
It is manually started with no logon trigger or automatic restart.

The planned cells are low/balanced, low/imbalanced, high/balanced, and
high/imbalanced. The existing 200-scenes-per-cell dataset is validated by
`mosaic_toy_train.load_cells`, including the 50:50 and 90:10 color quotas and
image hashes. Each cell trains a fresh PilotUNet with width 16, batch 4, 1,000
timesteps, velocity objective, AdamW, identical initial seed 81570 and a
150,000-update budget. After the initial 1,000-update probe, the continuation
restores model, EMA and optimizer state and starts a new matched RNG stream with
seed 81571. This transition is **not an exact resume**, because the initial
probe does not save RNG state. Later checkpoints include optimizer and RNG state
and are written at cumulative updates 10,000, 20,000, ..., 150,000. The four
cells run sequentially; cell directories and snapshots are never overwritten.

This is one matched training-seed block on a small fixed dataset. It does not
itself establish the MOSAIC direction, a reliable uncertainty interval, the
checker gate, or Stage 0. It is an adaptation, not the published MOSAIC setup.
Predefine evaluation and use independent training seed blocks before drawing a
research conclusion. In particular, existing checker abstentions and selected
samples cannot be treated as failures or successes. The use of identical seeds
pairs initialization and random streams; it does not create independent runs.

The high/imbalanced wrapper and its checkpointed continuation completed a
one-update-per-stage CUDA smoke test; restored model, EMA and optimizer hashes
matched, and the continuation saved RNG state. A focused cell-selection test
passed on the laptop. The scheduled task was observed Running on the RTX 3050;
the low/balanced initial run reached 1,000 updates, wrote its report and
checkpoint, and the continuation wrote provenance and started. No 10,000-update
checkpoint was observed at launch time. An earlier laptop run took roughly
824 seconds per 10,000 updates; extrapolating gives about 14 hours for 600,000
updates, plus previews and startup. Actual duration can differ substantially.

On the laptop, inspect with:

```powershell
Get-ScheduledTask MuadibMosaicMatched81570 | Select-Object TaskName, State
Get-Content "$env:USERPROFILE\muadib\mosaic-matched-81570.log" -Tail 40
Get-ChildItem "$env:USERPROFILE\muadib\mosaic-gpu-81300\outputs" -Directory -Filter 'mosaic-matched-81570-*'
```

The task requests that Windows avoid idle sleep, but cannot prevent lid-close
sleep, shutdown, restart, or forced sleep. An interruption stops the current
process. Preserve all partial outputs and validate the last complete checkpoint;
the one-shot runner refuses existing paths and does not resume automatically.
Later `warmstart-*.pt` checkpoints can be resumed only with the same selected
cell, dataset and compatible runtime after checking their integrity. Do not
restart the old continuous task as part of this experiment.
