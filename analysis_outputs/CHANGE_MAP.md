# Change Map: Modifiability & Safety Boundaries

This document defines conservative boundaries for modifying the codebase, distinguishing safe exploration zones from protected ground-truth verification infrastructure.

---

## 1. Allowed Change Zones (High Safety / Exploration)
These components are intended for iterative research development, model architecture exploration, and training tuning:
- [`exposure/pilot_train.py`](file:///home/mango/Repos/muadib/exposure/pilot_train.py): Backbone model architectures (`PilotUNet`), diffusion schedules, loss functions, optimizer setups.
- [`exposure/velocity_online.py`](file:///home/mango/Repos/muadib/exposure/velocity_online.py): Velocity / flow matching objectives and sampling algorithms.
- [`exposure/velocity_guidance.py`](file:///home/mango/Repos/muadib/exposure/velocity_guidance.py): Guidance formulations and inference-time guidance scaling.
- [`exposure/design_sample.py`](file:///home/mango/Repos/muadib/exposure/design_sample.py): Hit-and-run sampling algorithms and proposal distributions for weight exploration.
- [`exposure/design_chains.py`](file:///home/mango/Repos/muadib/exposure/design_chains.py): Multi-chain sampling and convergence diagnostics.
- [`exposure/design_bounds.py`](file:///home/mango/Repos/muadib/exposure/design_bounds.py): Exact LP exposure/lift bounds and positive-lift corner.
- [`exposure/benchmark_audit.py`](file:///home/mango/Repos/muadib/exposure/benchmark_audit.py): Adversarial benchmark-validity audit (detection only).
- Probes and diagnostics (`exposure/*_probe.py`, `exposure/*_diagnostic.py`): Diagnostic measurement scripts.

---

## 2. Protected Eval Zones (Strict Ground Truth)
Do not alter these files during modeling experiments; changes here risk breaking evaluation validity or modifying ground truth:
- [`exposure/rules.py`](file:///home/mango/Repos/muadib/exposure/rules.py): Defines core rule satisfiability semantics, existential quantification, and object bindings. Modifying this invalidates all ground-truth labels and historical comparisons. The 2026-11-05 change adds `validate_rule` only; `satisfies` semantics are unchanged.
- [`exposure/grammar_audit.py`](file:///home/mango/Repos/muadib/exposure/grammar_audit.py): Audits canonical grammar layouts and witnesses.
- [`tests/test_rules.py`](file:///home/mango/Repos/muadib/tests/test_rules.py), [`tests/test_conditioned.py`](file:///home/mango/Repos/muadib/tests/test_conditioned.py): Unit tests establishing formal semantics.
- [`exposure/audit_agreement.py`](file:///home/mango/Repos/muadib/exposure/audit_agreement.py): Pairwise raw agreement and unresolved disagreements.

---

## 3. Data Interface Zones
Modules that define schemas and data ingestion interfaces:
- [`exposure/render.py`](file:///home/mango/Repos/muadib/exposure/render.py) (`parse_rule`, `RenderedScene`): Defines JSON serialization format for rules and rendered scene annotations.
- [`exposure/measure.py`](file:///home/mango/Repos/muadib/exposure/measure.py) (`load_rules`, `unique_object`): JSON schema parsing, duplicate key rejection, and structural rule validation.
- [`exposure/composition.py`](file:///home/mango/Repos/muadib/exposure/composition.py): Composition-gap scales, boundary status, and the multiplicative-null check.
- [`exposure/rules.py`](file:///home/mango/Repos/muadib/exposure/rules.py) (`validate_rule`): Rejects duplicate predicates, contradictory same-variable attributes, and cyclic strict relations.
- [`exposure/design.py`](file:///home/mango/Repos/muadib/exposure/design.py) (`solve`): Schema for constraint lists (`min`, `max`, `rule`, `predicate_indices`).
- `examples/*.json`: Canonical benchmark rule sets and constraint specifications.

---

## 4. Single-Variable High-Risk Zones
Modifications in these areas have systemic effects across multiple pipeline stages:
- **Rule set specifications (`examples/exposure_rules.json`, `examples/templates.json`):** Any change requires re-measuring the entire Monte Carlo co-satisfaction matrix ($A$-matrix) and re-running LP feasibility.
- **Renderer geometry & coordinates (`exposure/render.py`):** Changing grid size, object radius, or placement jitter will alter co-satisfaction probabilities and break `exposure/checker.py` bounding-box heuristics.
- **Checker detection heuristics (`exposure/checker.py`):** Adjusting color distance thresholds or size criteria changes false positive/negative rates, directly biasing observed RSR.
