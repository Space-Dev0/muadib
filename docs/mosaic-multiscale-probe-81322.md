# Multiscale CUDA learnability probe

The preceding four-cell 5,000-update toy model produced four inspected previews
that looked like colored noise. Lower training loss alone did not establish
usable generation. This one-cell probe asks whether a larger multiscale
velocity network and a longer fixed budget produce recognizable samples; it
is exploratory and does not identify why the earlier samples failed.

The 2-level encoder-decoder adds a downsampled feature path and skip connection,
while retaining velocity-prediction MSE, the linear alpha schedule 0.01–0.99,
AdamW at 0.0001, batch 4, 64×64 data and the low/balanced cell of the exact
200-example dataset from `mosaic-gpu-feasibility-81302.md`. Width is 32. It
trains one model with seed 81322 and saves snapshots at 5,000 and 20,000
updates. At each snapshot it samples red and blue labels using identical noise
seeds 81330 and 81331, with 50 reverse steps and Gaussian starting noise. This
produces four previews per snapshot; they are not a representative estimate.
The model and source code differ from the 3-convolution baseline, and the
20,000-update model has more training than the prior model: comparisons do not
isolate an architectural effect. The alpha at the noisiest endpoint remains
0.01, so this probe does not evaluate a zero-terminal-SNR change.

`exposure/mosaic_multiscale_probe.py` records source and manifest hashes,
training settings, checkpoint hashes, selected loss summaries and PyTorch
version in `report.json` after training. The model checkpoints are probe-only
weights, with no optimizer state, and are not resumable. The output directory
is exclusive and a failure can leave partial results. Launch on the GPU laptop
using `scripts/run-mosaic-multiscale-probe.ps1`; the one-off scheduled task is
`MuadibMosaicMultiscale81322` and its log is
`~/muadib/mosaic-multiscale-probe-81322.log`. The data/code base is an isolated
copy of tracked files at `0f80a79`, plus the new probe module; the existing
800k model and continuous trainer are unaffected.

This is one cell, one dataset and one training seed. It is not a MOSAIC
replication, cannot estimate complexity-versus-imbalance effects, and cannot
establish Stage 0. A valid contrast needs independently trained models in all
four cells, multiple seed blocks, a fixed evaluation protocol and a validated
checker or human labels. Ideas motivating the exploratory intervention include
the multiscale U-Net used in [DDPM](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
and the [terminal-noise schedule analysis](https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf).
Neither paper establishes the cause of this project's generated noise.

The four 5,000-update previews were copied to ignored
`outputs/mosaic-multiscale-preview-81322/`. Informal visual inspection found
colored clusters and speckling, with no consistently recognizable clean shapes.
The standard checker abstained on all four for unrecognized foreground colors;
that checker rejection is not a human error label or a learnability metric.
The 20,000-update snapshot completed and its four paired previews were also
copied to the same directory. Informal inspection showed more coherent colored
block-like regions and spatial grouping than at 5,000 updates, with substantial
overlap and distortion still present. The standard checker abstained on all four
for unrecognized foreground colors. These eight previews are not independent
validation, human ratings, or estimates of rule satisfaction. The mean training
loss over the final 500 updates was 0.06714 at 5,000 and 0.02225 at 20,000;
lower loss does not establish generation quality. Recorded training elapsed time
was 26.6 seconds and 103.8 seconds respectively, excluding task startup.

## Exploratory 2×2 extension

The remaining low-imbalanced, high-balanced and high-imbalanced cells are
scheduled sequentially by `scripts/run-mosaic-multiscale-remaining.ps1`, task
`MuadibMosaicMultiscaleRemaining81322`. Each uses the same dataset manifest,
training seed 81322, width 32, batch size 4, optimizer, schedule, 20,000 update
budget and sampling labels/seeds as the completed low-balanced cell. This is a
single matched training-seed block and **not** a replicated experiment.

The initial low-balanced code hard-coded its cell. Before the extension the
original source was retrieved to ignored
`outputs/mosaic-multiscale-preview-81322/source-81322.py`; its SHA-256 matches
that cell's report. The new source exposes the cell in the CLI and checkpoint
metadata, changing its source hash. A direct source diff confirms the network,
training loop, initialization and sampler otherwise match. This source-version
difference remains part of provenance and must not be presented as an exact
single-source run. New cell outputs each have exclusive paths beginning
`outputs/mosaic-multiscale-81322-` on the GPU laptop. Completion and evaluation
are separate steps; task scheduling is not evidence that training finished.

The exploratory blinded preview-review plan was written after seeing the four
low-balanced previews and before inspecting the other cells, so it is not a
fully prespecified confirmatory protocol. It includes all 16 previews from
20,000 updates (four cells × two requested colors × two paired noise seeds),
with all four reviewers independently answering three separate questions:
whether the requested-color circle is present, whether exactly two objects are
visible, and whether exactly three objects are visible. Reviewers may answer
`unsure`. These count questions characterize the images; the model is not
conditioned on the requested count or complexity. The source manifest and
frozen checker outputs are kept private. Human labels will be descriptive, not
an estimate with sufficient independent model seeds for the MOSAIC gate.

All three additional cells completed the bounded 20,000-update budget and their
four 20k previews were retrieved. Mean training loss over the last 500 updates
was 0.02121 low-imbalanced, 0.02636 high-balanced and 0.02694 high-imbalanced;
low-balanced was 0.02225. These values are not object accuracy and should not
be used to compare MOSAIC's effect direction.

The blinded packs were generated at ignored
`outputs/mosaic-multiscale-review-packs-81322/reviewer-1.zip` through
`reviewer-4.zip`. Each of the 16 images appears in each pack; only the ZIPs
should be shared, not `private/`. `private/provenance.json` records cell/label,
sample seeds and artifact hashes; `private/predictions.json` contains frozen
checker outputs. Both standard and experimental checkers abstained on all 16
requested-color-circle judgments. That describes checker coverage only; it is
not evidence that all 16 human judgments are negative. Return the four filled
`reviews.csv` files separately; preserve `yes`, `no`, and `unsure` entries.

All four 20,000-update checkpoint files were also copied to their cell folders
under ignored `outputs/mosaic-multiscale-review-source-81322/`; each local SHA-256
matches its run report. The 5,000-update and earlier toy checkpoints remain on
the laptop.
