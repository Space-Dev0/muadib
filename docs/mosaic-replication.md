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

The RTX 3050 laptop has since run exploratory CUDA training. CPU-only
instrument checks do not count as the requested replication.

## Preparing a run-level contrast

`exposure.mosaic_contrast` provides a **provisional descriptive analysis** for
an adapted 2×2 design. It does not train models, generate scenes, validate a
checker or determine whether the MOSAIC gate passed. Before training, specify
the simple and complex scene conditions, how the class-frequency imbalance is
constructed, a common held-out evaluation distribution, seed pairing,
dataset size, training budget, guidance, accuracy metric and an uncertainty
method. The experiment must measure the intended complexity and skew rather
than confounding them with other changes. Record failed runs rather than
silently dropping them.

The input is a JSON object with a `runs` list. Each row represents a separately
trained model and contains a unique string `run_id`, nonnegative integer `seed`,
`complexity` (`low` or `high`), `imbalance` (`balanced` or `imbalanced`), and
integer `successes` and `total` for its evaluation. Each seed must have one
distinct model in each of the four cells. Equal seeds designate planned blocks;
they do not by themselves prove that the models or errors are independent.

```sh
uv run python -m exposure.mosaic_contrast --input outputs/mosaic-runs.json --output outputs/mosaic-contrast.json
```

Within each seed, the script computes per-model success proportions and averages
the low-minus-high complexity drop across both imbalance conditions. It also
averages the balanced-minus-imbalanced drop across both complexity conditions.
The difference between these drops is positive when the former is larger. Each
seed and cell receives equal weight, even when image totals differ; the report
retains the input counts. Images from one model are not independent training
replicates. This summary has no uncertainty interval or automatic go/no-go;
the full protocol, uncertainty and direction comparison remain to be specified
and evaluated on real runs. No such run data exist yet.
