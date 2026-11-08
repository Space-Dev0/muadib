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
- [Grammar audit](docs/grammar-audit.md): canonical rule layouts and witnesses.
- [Candidate templates](docs/templates.md) and [vocabulary coverage](docs/template-balance.md):
  provisional rule set and per-cell representation.
- [Prototype image checker](docs/checker.md): clean-render audit and limitations.
- [Checker perturbations](docs/checker-perturb.md): detector coverage and errors
  under controlled image changes.
- [Human audit workflow](docs/human-audit.md): prepare blinded reviews and score
  human judgments once generated images are available.
- [Human checklist for Stage 0](docs/human-checklist.md): human decisions,
  access, image review, GPU/MOSAIC work, and final gate sign-off.
- [Gaussian diagnostic](docs/gaussian-diagnostic.md): an analytic noise-loss
  reference for a one-dimensional Gaussian mixture.
- [Design diagnostics](docs/design-diagnostics.md): exploratory correlations
  and VIF for candidate designs.
- [Design subspace](docs/design-subspace.md): numerical singleton-preserving
  directions for an estimated exposure matrix.
- [Design sampler](docs/design-sample.md): exploratory hit-and-run sampling
  that preserves estimated singleton exposures.
- [Design bounds](docs/design-bounds.md): exact LP exposure and lift bounds,
  including the positive-lift corner.
- [Design chains](docs/design-chains.md): multi-chain split-R-hat and
  autocorrelation ESS diagnostics.
- [Benchmark audit](docs/benchmark-audit.md): adversarial relational-validity
  checks and `confirmatory_ready` status.
- [Composition gap](docs/composition-gap.md): the two composition-gap scales,
  boundary status, and the multiplicative-null check.
- [Mixture validation](docs/mixture-validation.md): compare matrix predictions
  against fresh sampled scenes.
- [Mixture analysis](docs/mixture-analysis.md): weighted exposures, rule and
  pair log lift, weight validation, and interpretation limits.
- [Exposure design](docs/exposure-design.md): feasibility constraints, LP
  formulation, solver statuses, and numerical limits.
- [Matched-marginal pilot](docs/matched-marginal-pilot.md): illustrative low and
  high full-rule exposure with two matched predicate marginals.
- [MOSAIC replication notes](docs/mosaic-replication.md): source review and
  outstanding prerequisites for the Stage 0 comparison.
- [Stage 0 status](docs/stage0-status.md): current instrument evidence and
  outstanding go/no-go requirements.
- [Architecture](docs/architecture.md): module interfaces, design decisions,
  current limits, and the test command.
- [Experiment specification](docs/experiment-spec.md): current rule and
  exposure semantics, unresolved decisions, and Stage 0 gates.
- [Research and hardware plan](docs/research-and-hardware-plan.md): dataset
  strategy, CPU/CUDA plans, and current limitations.
- [Pilot training](docs/pilot-training.md): exploratory online diffusion
  mechanics, commands, outputs, and limitations.
- [Windows GPU access](docs/windows-gpu-access.md): safe SSH access and the
  read-only GPU preflight for the reported RTX 3050 laptop.
