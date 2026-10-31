# Getting Started

This project implements the research proposal in stages. The current stage
evaluates rules on scene annotations and renders simple synthetic scenes,
optionally conditioned on a rule. It does not yet train a model, download an
external dataset, or provide CUDA acceleration.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Pillow, installed by `uv sync`

The current checks and renderer run on the CPU only.

## Install dependencies

From the repository root:

```sh
uv sync
```

## Run the checks

```sh
uv run python -m unittest discover -s tests -v
```

The tests cover rule evaluation, scene rendering, conditioned sampling, and the
command-line interface.

Rule evaluation uses distinct object assignments and existential rule
semantics. Coordinates follow the image convention: larger `y` means lower on
the image. `left_of` and `above` compare object centers strictly, so ties do not
satisfy a relation. Pair and marginal exposure semantics need to be locked in
the experiment specification before analysis.

## Render a first scene

```sh
uv run python -m exposure.render --seed 42 --output outputs/example
```

This writes `outputs/example.png` and `outputs/example.json`. The renderer uses
Pillow on the CPU and draws three separated shapes with antialiased edges. Each
annotation records the object's color, shape, and center coordinates.

Seeded output is reproducible when the code and dependency versions match.

## Next steps

- [Rendering and rules](rendering-and-rules.md) explains rule files, predicate
  semantics, conditioned sampling, and the output JSON.
- [Research and hardware plan](research-and-hardware-plan.md) covers the dataset
  strategy and the planned training hardware.
