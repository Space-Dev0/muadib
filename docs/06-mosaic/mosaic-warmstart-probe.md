# Multiscale weight-only warm-start probe

This bounded experiment tests whether 30,000 more updates improve a single
low-complexity balanced 20,000-update model. It is a **new optimizer run**,
not exact continuation: the original weight checkpoint contains neither AdamW
state nor RNG state. AdamW and random streams restart, with seed 81350, so any
change may reflect optimizer reset as well as additional updates. Only one cell
and training seed are tested; this is not a MOSAIC replication or Stage 0 gate.

`exposure.mosaic_warmstart_probe` verifies that the original checkpoint is a
20,000-update low-balanced model with the same dataset manifest and uses the
same network, batch size 4, AdamW learning rate 0.0001, alpha schedule and
velocity target. It saves weight-only snapshots after 10,000, 20,000 and 30,000 new
updates, and generates red/blue previews at the same sample seeds 81330 and
81331 and 50 sampler steps as the parent. Those seeds and visual comparisons
are exploratory and do not measure generalization. Snapshot names are
`warmstart-plus-10000.pt`, `warmstart-plus-20000.pt` and `warmstart-plus-30000.pt` to avoid misrepresenting
these as exact-resume checkpoints. The snapshots themselves do not
contain optimizer or RNG state and cannot be resumed exactly either.

The script records parent checkpoint and manifest hashes, its source hash,
PyTorch and Python versions, sampler seeds and loss summaries in `report.json`.
The output directory must not already exist; interrupted runs leave partial
artifacts which must not be silently overwritten. The default bounded Windows
command is `scripts/run-mosaic-warmstart-probe.ps1`, which expects the isolated
GPU dataset and initial checkpoint from the previous multiscale experiment.

Local verification: two focused unit tests and a one-update CPU smoke against
the real 20k checkpoint and dataset passed. That smoke provides no evidence of
sample quality. The bounded GPU run and its results are described below.

## Bounded CUDA result

The one-off task `MuadibMosaicWarmStart81350` completed the full 30,000-new-update
budget and is no longer running. It used the RTX 3050 and PyTorch 2.9.1+cu128,
with outputs copied to ignored `outputs/mosaic-warmstart-probe-81350/`. The
report and all three local checkpoint hashes match. Last-500-update mean losses
were 0.01763, 0.01469 and 0.01341 at +10k, +20k and +30k updates respectively.
On informal inspection of the paired red 81330 and blue 81331 previews, the
+30k samples appear less speckled and more coherent than the 20k parent, though
shapes still merge and distort. Two seeds and one training cell cannot establish
an accuracy gain or benchmark behavior. No independent human ratings were used.
The GPU and local `mosaic_multiscale_probe.py` implementation hashes were
verified equal after the run:
`c3bf20b769959871df6fd9be9ccc0d41ec109445a45ad191ef2fd5b755fa7f51`.
This includes the model and sampler implementation; it differs from the
original parent checkpoint's earlier source hash because the parent used a
hard-coded training cell.
