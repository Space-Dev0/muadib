# Which Exposure Matters?

This project implements the research proposal in stages. It currently evaluates
rules on annotations and renders simple synthetic scenes, optionally conditioned
on a rule. It does not yet train a model, download an external dataset, or
provide CUDA acceleration. The renderer is a CPU-only prototype, not the
validated exposure design.

## Quick start

With Python 3.11+ and uv installed, from the repository root:

```sh
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m exposure.render --seed 42 --rule examples/red_left_of_square.json --output outputs/example
```

`uv sync` installs dependencies, the second command runs the test suite, and the
third renders a rule-conditioned scene to `outputs/example.png` and
`outputs/example.json`. Omit `--rule` to render without a condition.

## Documentation

- [Getting started](docs/getting-started.md): setup, tests, and a first render.
- [Rendering and rules](docs/rendering-and-rules.md): command-line options, rule
  file format, predicate semantics, sampling, and output JSON.
- [Exposure measurement](docs/exposure-measurement.md): named rule sets, the
  measurement CLI, satisfaction columns, Wilson intervals, and limitations.
- [Mixture analysis](docs/mixture-analysis.md): weighted exposures, rule and
  pair log lift, weight validation, and interpretation limits.
- [Exposure design](docs/exposure-design.md): feasibility constraints, LP
  formulation, solver statuses, and numerical limits.
- [Architecture](docs/architecture.md): module interfaces, design decisions,
  current limits, and the test command.
- [Research and hardware plan](docs/research-and-hardware-plan.md): dataset
  strategy, CPU/CUDA plans, and current limitations.
