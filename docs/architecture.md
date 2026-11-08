# Architecture

This repository is a staged, CPU-only prototype for studying exposure in
conditional generative models. It currently evaluates rules on scene
annotations, renders synthetic scenes, measures rule satisfaction, summarizes
mixtures, and checks Stage 0 exposure feasibility on an estimated matrix. It
does not train a model, download an external dataset, or provide CUDA
acceleration.

## Modules and interfaces

| Module | Interface and role |
| --- | --- |
| `exposure/rules.py` | `SceneObject`, `Attribute`, `Relation`, `satisfies`, and `validate_rule`. Evaluates rule conjunctions with distinct object bindings and existential semantics; rejects duplicate, contradictory, and cyclic rules. |
| `exposure/render.py` | `render_scene`, `render_conditioned`, `sample_conditioned`, `parse_rule`, and the render CLI. Samples three-object grid scenes and renders 64x64 PNGs with JSON annotations. |
| `exposure/measure.py` | `load_rules`, `estimate`, `wilson_interval`, and the measurement CLI. Produces schema-1 reports, optionally with singleton and pair `predicate_subsets`. |
| `exposure/mixture.py` | `summarize` and the mixture CLI. Validates subset-enabled reports, computes weighted rule and pair exposures, and reports log lift plus both composition gaps and their boundary status. |
| `exposure/composition.py` | `logit`, `gamma_logit`, `gamma_log`, `composition_gaps`, and `null_gaps`. Evaluates the two composition-gap scales and their multiplicative null. |
| `exposure/design.py` | `solve` and the design CLI. Validates a report, checks feasibility with a continuous LP, and optionally minimizes KL divergence to uniform with SLSQP. |
| `exposure/design_subspace.py` | SVD analysis of estimated singleton-preserving equality directions. |
| `exposure/design_sample.py` | Seeded hit-and-run sampling inside the equality nullspace. |
| `exposure/design_bounds.py` | Exact LP exposure and lift bounds, boundary status, and the positive-lift corner. |
| `exposure/design_chains.py` | Multi-chain hit-and-run with split-R-hat and autocorrelation ESS diagnostics. |
| `exposure/grammar_audit.py` | Audits canonical rule layouts and constructs witnesses for the prototype grammar. |
| `exposure/templates.py` | Builds a provisional 57-template rule set and manifest. |
| `exposure/template_balance.py` | Reports per-cell vocabulary coverage without declaring balance. |
| `exposure/benchmark_audit.py` | Adversarial relational-validity audit: saturation, endpoint binding, same-object pair coverage, held-out split, and `confirmatory_ready`. |
| `exposure/validate_mixture.py` | Compares matrix predictions with independently sampled mixture scenes. |
| `exposure/checker.py` | Prototype pixel-based object and rule checker with a clean-render audit and per-rule conditional/unconditional detection metrics. |
| `exposure/checker_perturb.py` | Controlled brightness and blur audits, including rejected detections, coverage, and per-rule metrics. |
| `exposure/human_audit.py` | Prepares blinded stratified reviews and scores actual human judgments. |
| `exposure/gaussian_diagnostic.py` | One-dimensional Gaussian-mixture Bayes denoiser and noise-loss reference. |
| `exposure/design_diagnostics.py` | Exploratory pooled correlation and VIF diagnostics for candidate allocations. |
| `exposure/design_subspace.py` | SVD analysis of estimated singleton-preserving equality directions. |
| `examples/` | Example rule sets, weights, and loose feasibility constraints. |
| `tests/` | Unit and CLI tests for rules, rendering, measurement, mixtures, and design feasibility. |

## Decisions

- Rule evaluation uses distinct object assignments, shared variable bindings
  within a rule or subset, and existential quantification.
- Scene sampling uses bounded rejection from a three-object grid distribution.
  Failure within the budget does not prove infeasibility.
- Measurement and mixture results are Monte Carlo plug-in estimates. Measurement
  Wilson intervals are pointwise, and mixture analysis does not propagate them.
- Design feasibility is solved as a continuous simplex LP:
  `sum(w) = 1`, `w >= 0`, and the requested rule/predicate/pair bounds. There
  are no integer decisions.
- `exposure.design` validates the report with uniform weights before solving,
  but the uniform weights are only a validation device, not the result.
- Only the infeasible solver status is reported as infeasible. Other solver
  failures are surfaced as errors.
- Minimum-KL allocation is a numerical optimization of the estimated matrix;
  it has no exact optimality certificate.

## Current limits

- No trained model, externally validated image checker, external dataset, CUDA
  acceleration, or validated exposure design is implemented.
- Estimates are finite-sample and descriptive; uncertainty is not propagated,
  and fixed marginals require explicit equality bounds.
- LP feasibility applies only to the estimated matrix. It is not population
  feasibility, and zero empirical counts are not proof of impossibility.
- The renderer and all analysis run on the CPU only.

## Test command

```sh
uv run --no-sync python -m unittest discover -s tests -q
```

See [Exposure design](exposure-design.md) for the constraint schema and LP
details.
