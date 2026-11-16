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

**Start with the [documentation guide](docs/README.md).** It explains how the docs
are organized and gives reading paths for presenters, engineers, and reviewers.

- [Getting started](docs/00-start-here/getting-started.md) and
  [architecture](docs/00-start-here/architecture.md): setup, tests, modules, and limits.
- [Experiment specification](docs/00-start-here/experiment-spec.md): rules, exposure,
  composition gaps, and pending decisions.
- [Core instruments](docs/01-instruments/): rendering, rules, exposure measurement,
  mixture analysis, and composition gaps.
- [Benchmark](docs/02-benchmark/): the provisional 57-rule set and its validity audit.
- [Exposure design](docs/03-exposure-design/): LP feasibility, sampling, bounds, and
  decorrelation diagnostics.
- [Checker and human audit](docs/04-checker-and-human-audit/): pixel checker, perturbation
  audits, and blinded review workflow.
- [Training](docs/05-training/pilot-training.md): exploratory diffusion pilot, probes, and
  measured limitations.
- [MOSAIC-inspired work](docs/06-mosaic/): source review, feasibility runs, and the
  four-cell matched experiment.
- [Status and reviews](docs/07-status-and-reviews/stage0-status.md): the authoritative
  Stage 0 gate dashboard and internal reviews.
- [Operations](docs/08-operations/): Windows GPU access and throughput benchmarking.

For the full narrative, code walkthrough, and presentation script, see
[`presentation/PROJECT-WALKTHROUGH.md`](presentation/PROJECT-WALKTHROUGH.md).
