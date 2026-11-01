# Research and Hardware Plan

The project implements the research proposal in stages. The current stage is a
CPU-only prototype for rendering and rule evaluation, plus an exploratory pilot
training instrument. No actual CUDA model training has been run, no external
dataset is downloaded, and the exposure design is not yet validated.

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
installed. CUDA availability through PyTorch, memory fit, and available GPU time
remain unverified. The read-only preflight itself installed and trained nothing.
Since then, `uv` has been installed in an isolated venv on the remote machine
and the CUDA PyTorch download is in progress. GPU fit remains unverified. The
repository has not been transferred; transfer awaits explicit authorization. The
correct PyTorch build has not been verified, and OpenCL is not part of this
implementation.

The optional training dependency is locked; CUDA execution and memory fit must
be verified when the training environment is ready.

## Pilot training

`exposure/pilot_train.py` is an exploratory online conditional diffusion
instrument; see [pilot training](pilot-training.md) for commands and limits. It
uses the online renderer, predicate-slot cross-attention, classifier-free
dropout, AdamW, EMA, and a fixed sampler. A separate CPU-only PyTorch 2.9.1
environment passed the CPU unit tests and a one-step train/checkpoint/sample CLI
smoke; these are mechanics only, not learning evidence. No actual CUDA training,
validated model outputs, reference model, resume, controlled guidance sweep, or
human audit exists. No mixed precision is implemented.

## Human review capacity

Four team members have been reported as potential image reviewers available
later. Their independence, blinding, and actual judgments are not yet
established, and no reviews have been completed.

## Current limitations

- Rendering and rule evaluation run on the CPU only.
- Exploratory pilot training code exists, but no actual CUDA model training has
  been run.
- No external dataset is downloaded or bundled.
- The read-only GPU preflight succeeded, but CUDA through PyTorch, memory fit,
  and available GPU time are unverified.
- The remote `uv` environment is being prepared and the CUDA PyTorch download is
  in progress; GPU fit remains unverified.
- No packages were installed during the preflight; the separate `uv` setup is in
  progress.
- The repository has not been transferred; transfer awaits explicit
  authorization.
- No validated model outputs, reference model, resume, controlled guidance
  sweep, or human audit exists.
- No image reviews have been completed.
- Pair and marginal exposure semantics must be locked in the experiment
  specification before analysis.
