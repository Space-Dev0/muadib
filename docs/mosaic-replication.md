# MOSAIC replication: preliminary source review

The proposal asks for a small replication of MOSAIC's complexity-versus-
imbalance contrast as a Stage 0 gate. This review is preparation only: **no
MOSAIC experiment or replication has been run**.

The [MOSAIC paper](https://proceedings.mlr.press/v306/jeong26b.html) reports
that scene complexity dominated concept imbalance in its studied low-data
regimes. Its [released code](https://github.com/eugene6923/MOSAIC) describes
basic and complex scene variants, held-out compositional combinations, Blender
dataset generation, model training, and classifier-based evaluation. The code's
example training command uses four GPUs, and the evaluation workflow requires
task-specific pretrained classifiers and generated or downloaded data.

The current project uses a different 2D renderer and an experimental
image-based checker. Reproducing a *direction* on this renderer would therefore
be an adapted contrast, not a full reproduction of MOSAIC's pipeline.

In the [paper's experimental setup](https://arxiv.org/html/2605.00273), the
complexity variants add objects or distractors; controlled skew changes class
frequencies at several dataset sizes. Its diffusion backbone operates in a
frozen pretrained VAE's latent space, and task-specific classifiers evaluate
generated images. Our prototype instead uses three 2D objects, currently has
no trained model, and has not validated its checker on generated outputs. These
differences must be controlled or explicitly reported in any comparison.

Before running or claiming this gate:

1. Read the full MOSAIC paper and relevant implementation, and specify the
   contrast, metrics, dataset sizes and balancing conditions to reproduce.
2. Identify a feasible GPU budget and whether its data and classifier weights
   are available for the selected workflow. Check their licenses and versions.
3. Define how this project's 2D adaptation differs, including the task,
   complexity manipulation, conditioning, training budget, and checker.
4. Run and record seeds, configurations, code versions and results. Compare the
   direction with MOSAIC, including uncertainty and failed runs.

The current environment did not expose `nvidia-smi` during the preliminary
check. Actual capabilities should be checked on the training machine; CPU-only
instrument checks do not count as the requested replication.
