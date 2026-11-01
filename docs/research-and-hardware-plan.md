# Research and Hardware Plan

The project implements the research proposal in stages. The current stage is a
CPU prototype for rendering and rule evaluation, plus an exploratory pilot
training instrument. A CUDA environment is installed on the prepared laptop,
but no project pilot model has been trained on CUDA, no external dataset is
downloaded, and the exposure design is not yet validated.

## Dataset plan

The proposal's primary data source is a renderer generating annotated synthetic
scenes online. A finite synthetic dataset is a secondary experiment.

An external dataset is not selected yet. Its annotations, license, condition
labels, and feasibility for controlled exposures must be assessed first.

Gomez et al.'s DL4IVF embryo time-lapse data is a possible separate follow-up,
not selected for the primary experiment. Its published stage labels alone do
not supply the object-level relational annotations needed for the proposed
synthetic study. See the
[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9120221/) and the
[dataset](https://zenodo.org/records/7912264/).

## Hardware plan

The planned training backend is PyTorch on NVIDIA CUDA, with CPU support for
small local checks. A user-authorized, read-only SSH GPU preflight succeeded on
the reported laptop. It reported Windows 11 AMD64, Python 3.13.6, an NVIDIA
GeForce RTX 3050 6GB Laptop GPU with 6144 MiB and driver 581.83, Git 2.48.1,
and pip 25.2. At preflight time, `uv` was not on PATH and PyTorch was not
installed. The isolated environment is now complete: `uv` 0.13.6, PyTorch
2.9.1+cu128 with CUDA 12.8, NumPy 2.5.4, Pillow 13.0.0, and SciPy 1.18.1, with
`torch.cuda.is_available()` true.

A generic CUDA smoke completed with a two-layer 64×64 synthetic convolution
model: two warmup steps and three timed CUDA forward/backward/AdamW steps, a
finite loss, 0.0695 seconds measured time, 1,804,288 bytes peak allocated, and
4,194,304 bytes peak reserved. This is a generic CUDA smoke only; it is not
project pilot fit, learning evidence, Stage 0 passage, or a study. Available GPU
time remains unverified.

The repository has not been transferred to the laptop. OpenCL is not part of
this implementation.

The optional training dependency is locked; CUDA execution is verified for the
generic smoke, while project model fit and GPU time remain to be measured.

## Pilot training

`exposure/pilot_train.py` is an exploratory online conditional diffusion
instrument; see [pilot training](pilot-training.md) for commands and limits. It
uses the online renderer, predicate-slot cross-attention, classifier-free
dropout, AdamW, EMA, and a fixed sampler. CPU pilot tests and the one-step
train/checkpoint/sample mechanical sample passed in a separate CPU PyTorch 2.9.1
environment; these are mechanics only, not learning evidence. The laptop CUDA
environment is installed and a generic CUDA smoke completed, but no project
pilot training has run on CUDA. No validated model outputs, reference model,
resume, controlled guidance sweep, or human audit exists. No mixed precision is
implemented.

## Human review capacity

Four team members have been reported as potential image reviewers available
later. Their independence, blinding, and actual judgments are not yet
established, and no reviews have been completed.

## Current limitations

- Rendering and rule evaluation run on the CPU only.
- Exploratory pilot training code exists, but no project pilot model has been
  trained on CUDA.
- No external dataset is downloaded or bundled.
- The isolated laptop environment is installed and `torch.cuda.is_available()`
  is true; a generic CUDA smoke completed. Project pilot fit and available GPU
  time remain unverified.
- The preflight itself installed nothing; the later isolated-venv installation
  is complete.
- The repository has not been transferred to the laptop.
- No validated model outputs, reference model, resume, controlled guidance
  sweep, or human audit exists.
- No image reviews have been completed.
- Pair and marginal exposure semantics must be locked in the experiment
  specification before analysis.
