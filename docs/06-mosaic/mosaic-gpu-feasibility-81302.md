# Bounded CUDA MOSAIC-inspired feasibility run

This experiment tests whether a larger toy dataset and GPU training produce
recognizable samples, before investing in a replicated contrast. It is **not**
MOSAIC, a validated estimator, or a Stage 0 gate result. Its 2D renderer,
architecture, training objective and image evaluator differ from MOSAIC.

The dataset was generated locally before training:

```sh
uv run python -m exposure.mosaic_pilot_data --seed 81300 \
  --train-per-cell 200 --eval-per-complexity 40 \
  --output outputs/mosaic-data-gpu-feasibility-81300
```

There are 200 images in each of four training cells and 80 held-out images.
Low/high complexity changes the number of distractors. Balanced/imbalanced
changes the red:blue target-class ratio from 50:50 to 90:10. The shared held-out
set is balanced by complexity and target class. Training cells have the same
number of examples but use distinct samples. Dataset generation checks for
pixel-array duplicates; see `mosaic-data-feasibility.md` for semantics.

An isolated copy of tracked `exposure` files at Git revision
`0f80a79abe1e4ff2ed78d9f3d72e4ab7a6261b5a` and the dataset was staged on the
GPU laptop at `~/muadib/mosaic-gpu-81300`. PyTorch 2.9.1+cu128 detected an
NVIDIA GeForce RTX 3050 Laptop GPU, and a 50-update-per-cell CUDA smoke run
completed. `scripts/run-mosaic-gpu-feasibility.ps1` starts one bounded run:
four separate toy velocity models, seed 81302, 5,000 updates per model,
batch size 4, width 16, then two paired sampler seeds for each cell and label
(16 exploratory images). The script uses new output paths, rejects existing
outputs and does not modify the existing 800k checkpoint or continuous trainer.

The one-off Windows scheduled task `MuadibMosaicFeasibility81302` starts the
script without a recurring trigger. Logs are at
`~/muadib/mosaic-gpu-feasibility-81302.log`. Checkpoints are non-resumable and
one model checkpoint is saved only after that cell completes; interruption may
leave partial outputs which must be preserved and not silently treated as a
complete run. The experiment records configuration, loss traces, dataset and
trainer hashes in `outputs/mosaic-gpu-feasibility-81302.json` when all four
training cells finish. Samples are only previews, not a representative evaluation.

A valid comparison still needs learnable generation, a separately validated
checker or blinded human evaluation, a prespecified success metric and contrast,
multiple independently trained seed blocks and uncertainty. The tiny network,
200-image cells, and one training block mean this run can only reveal immediate
feasibility problems; it cannot decide whether scene complexity dominates class
imbalance.

## Completed run

All four cells completed 5,000 CUDA updates and 16 PNG previews were retrieved
locally with the training report. The mean loss over the first 500 steps was
0.613, 0.612, 0.621 and 0.622 for low-balanced, low-imbalanced, high-balanced
and high-imbalanced respectively; over the last 500 steps it was 0.105, 0.103,
0.119 and 0.121. These are training losses, not generation success or a
replication comparison. Four individual previews (one from each cell) looked
like colored noise on visual inspection. Four images are insufficient for a
population estimate, and no human audit has been performed. The results show
that this configuration does not yet provide obvious usable evidence of
learnability; they do not show longer training or a different architecture
cannot work.

The report and preview images are saved in ignored local paths
`outputs/mosaic-gpu-feasibility-81302.json` and
`outputs/mosaic-gpu-feasibility-samples-81302/`. Model checkpoints remain on
the GPU laptop under the checkpoint directory specified in the script.

## Result and research leads

The one-off task completed all four cells and 16 previews. First-500 versus
last-500 mean training losses decreased from approximately 0.61–0.62 to
0.102–0.121, but inspection of four previews (one per cell) found colored
noise rather than recognizable scenes. This is not an accuracy measurement or
proof that the task cannot be learned. The image samples and report were copied
locally under `outputs/mosaic-gpu-feasibility-samples-81302/` and
`outputs/mosaic-gpu-feasibility-81302.json`.

The toy denoiser uses three 3×3 convolutions, giving a seven-pixel receptive
field per denoiser call, and its schedule ends at alpha=0.01 while sampling
starts from Gaussian noise. These are testable limitations, not established
causes of failure. The original [DDPM paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
uses a multiscale U-Net with attention. [Lin et al.](https://openaccess.thecvf.com/content/WACV2024/papers/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.pdf)
identify a possible mismatch when training retains terminal signal but sampling
starts from noise; their result does not prove that mismatch explains this run.
A controlled architecture/schedule comparison and recognizable generation are
needed before proceeding to a meaningful MOSAIC-style contrast.

An independent OpenCode review on the GPU laptop recommended checking the
velocity/sampler contract with oracle inversion, measuring error by noise level,
performing fixed-image overfit and conditioning ablations, and comparing full
sampling under matched schedules before attributing failure to capacity. The
existing one-step and multistep oracle diagnostics cover parts of that list;
they do not by themselves prove the learned sampler is correct. The multiscale
probe is exploratory rather than an isolated causal test of architecture.
