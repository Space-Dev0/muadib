# Which Exposure Matters?

This project implements the research proposal in stages. It currently evaluates
rules on annotations and renders simple synthetic scenes. It does not yet train
a model, download an external dataset, or provide CUDA acceleration.

## Run the first checks

With Python 3.11+ and uv installed, from this directory:

```sh
uv sync
uv run python -m unittest discover -s tests -v
```

The checks run on a CPU. Rule evaluation uses
distinct object assignments, existential rule semantics, and image coordinates
where larger y means lower on the image. `left_of` and `above` compare object
centers strictly, so ties do not satisfy a relation. Pair and marginal exposure
semantics need to be locked in the experiment specification before analysis.

## Render a scene

```sh
uv run python -m exposure.render --seed 42 --output outputs/example
```

This creates `outputs/example.png` and `outputs/example.json`. The renderer
uses Pillow on the CPU, draws three separated shapes with antialiased edges,
and records color, shape, and center coordinates for each object. The seed
reproduces the image and annotations with the same dependency versions. This
prototype samples objects independently; it does not yet generate scenes for
a requested rule, build masks, or implement the exposure design.

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
