# Which Exposure Matters?

This project implements the research proposal in stages. The first stage defines
how rules are evaluated against ground-truth scene annotations. It does not yet
render images, train a model, download a dataset, or provide CUDA acceleration.

## Run the first checks

With Python 3.11+ and uv installed, from this directory:

```sh
uv run --no-sync python -m unittest discover -s tests -v
```

The first step has no third-party dependencies and runs on a CPU. It uses
distinct object assignments, existential rule semantics, and image coordinates
where larger y means lower on the image. `left_of` and `above` compare object
centers strictly, so ties do not satisfy a relation. Pair and marginal exposure
semantics need to be locked in the experiment specification before analysis.

## Dataset and hardware plan

The proposal's primary data source is a renderer generating annotated synthetic
scenes online. A finite synthetic dataset is a secondary experiment. An external
dataset is not selected yet; its annotations, license, condition labels, and
feasibility for controlled exposures must be assessed first.

The planned training backend is PyTorch on NVIDIA CUDA, with CPU support for
small local checks. The correct PyTorch build depends on the friend's GPU,
driver, operating system and Python version; it has not yet been selected or
installed. OpenCL is not part of this implementation. GPU support and a locked
dependency set will be added and verified when the training stage is built.
