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
small local checks. The correct PyTorch build depends on the friend's GPU,
driver, operating system, and Python version; it has not yet been selected or
installed. OpenCL is not part of this implementation.

GPU support and a locked dependency set will be added and verified when the
training stage is built.

## Current limitations

- Rendering and rule evaluation run on the CPU only.
- No model training is implemented.
- No external dataset is downloaded or bundled.
- No CUDA acceleration is available yet.
- Pair and marginal exposure semantics must be locked in the experiment
  specification before analysis.
