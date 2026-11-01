# Research and Hardware Plan

The project implements the research proposal in stages. The current stage is a
CPU-only prototype for rendering and rule evaluation. It does not yet train a
model, download an external dataset, or provide CUDA acceleration. It is not
yet the proposal's validated exposure design.

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
and pip 25.2. `uv` is not on PATH and PyTorch is not installed. CUDA
availability through PyTorch, memory fit, and available GPU time remain
unverified. No training or installation occurred. The correct PyTorch build has
not been selected or installed, and OpenCL is not part of this implementation.

GPU support and a locked dependency set will be added and verified when the
training environment is set up.

## Human review capacity

Four team members have been reported as potential image reviewers available
later. Their independence, blinding, and actual judgments are not yet
established, and no reviews have been completed.

## Current limitations

- Rendering and rule evaluation run on the CPU only.
- No model training is implemented.
- No external dataset is downloaded or bundled.
- No CUDA acceleration is available yet.
- The read-only GPU preflight succeeded, but CUDA through PyTorch, memory fit,
  and available GPU time are unverified.
- No packages were installed during the preflight, and no project training has
  been run on the reported machine.
- No image reviews have been completed.
- Pair and marginal exposure semantics must be locked in the experiment
  specification before analysis.
