# Research Map

- **Task Family:** `conditional_diffusion_exposure`
- **Repository Root:** `/home/mango/Repos/muadib`
- **README Present:** `True`

---

## 1. Entrypoints

### Train Entrypoints
- [`exposure/pilot_train.py`](file:///home/mango/Repos/muadib/exposure/pilot_train.py): Main conditional DDPM diffusion trainer (`python -m exposure.pilot_train train ...`).
- [`exposure/velocity_online.py`](file:///home/mango/Repos/muadib/exposure/velocity_online.py): Flow-matching / velocity prediction online trainer.
- [`exposure/high_t_finetune.py`](file:///home/mango/Repos/muadib/exposure/high_t_finetune.py): High-timestep noise loss fine-tuning.
- [`exposure/identity_eps_finetune.py`](file:///home/mango/Repos/muadib/exposure/identity_eps_finetune.py): Identity baseline diagnostic fine-tuning.

### Inference & Sampling Entrypoints
- [`exposure/pilot_train.py`](file:///home/mango/Repos/muadib/exposure/pilot_train.py): Sampling command (`python -m exposure.pilot_train sample ...`).
- [`exposure/velocity_guidance.py`](file:///home/mango/Repos/muadib/exposure/velocity_guidance.py): Classifier-free velocity sampler with guidance scaling.
- [`exposure/render.py`](file:///home/mango/Repos/muadib/exposure/render.py): Synthetic ground-truth conditional scene renderer.

### Evaluation & Audit Entrypoints
- [`exposure/measure.py`](file:///home/mango/Repos/muadib/exposure/measure.py): Co-satisfaction $A$-matrix measurement CLI.
- [`exposure/mixture.py`](file:///home/mango/Repos/muadib/exposure/mixture.py): Mixture exposure and log-lift calculator.
- [`exposure/validate_mixture.py`](file:///home/mango/Repos/muadib/exposure/validate_mixture.py): Monte Carlo validation of design mixtures.
- [`exposure/design.py`](file:///home/mango/Repos/muadib/exposure/design.py): Simplex LP exposure feasibility and min-KL solver.
- [`exposure/checker.py`](file:///home/mango/Repos/muadib/exposure/checker.py): Image-based rule checker CLI.
- [`exposure/checker_perturb.py`](file:///home/mango/Repos/muadib/exposure/checker_perturb.py): Checker perturbation robustness test.
- [`exposure/human_audit.py`](file:///home/mango/Repos/muadib/exposure/human_audit.py): Double-blinded human audit preparation and scoring.
- [`exposure/audit_agreement.py`](file:///home/mango/Repos/muadib/exposure/audit_agreement.py): Multi-annotator agreement and Cohen/Fleiss kappa.

### Model Definitions
- [`exposure/pilot_train.py`](file:///home/mango/Repos/muadib/exposure/pilot_train.py): `PilotUNet`, `Residual`, `SelfAttentionBlock`, `encode_rules`.
- [`exposure/velocity_probe.py`](file:///home/mango/Repos/muadib/exposure/velocity_probe.py): `VelocityAsNoise`, velocity conversion formulas.
- [`exposure/time_embedding_probe.py`](file:///home/mango/Repos/muadib/exposure/time_embedding_probe.py): Sinusoidal time embedding module.

### Configuration & Data Specifications
- `examples/exposure_rules.json`: Core rule dictionary.
- `examples/exposure_constraints.json`: Feasibility constraints specification.
- `examples/uniform_weights.json`: Base uniform sampling weights.
- `examples/fixed_marginals_low.json` & `examples/fixed_marginals_high.json`: Marginal-preserving comparison allocations.
- `examples/red_left_of_square.json`: Single rule condition example.

---

## 2. Task-Relevant Files & Documents

- [`docs/architecture.md`](file:///home/mango/Repos/muadib/docs/architecture.md): Architectural decisions and constraints.
- [`docs/stage0-status.md`](file:///home/mango/Repos/muadib/docs/stage0-status.md): Comprehensive review of Stage 0 instrument milestones.
- [`docs/review-2026-11-04.md`](file:///home/mango/Repos/muadib/docs/review-2026-11-04.md): Methodological and code review assessing proposal Rev. 4.
- [`docs/pilot-training.md`](file:///home/mango/Repos/muadib/docs/pilot-training.md): Online training mechanics and exploratory results.
- [`docs/exposure-design.md`](file:///home/mango/Repos/muadib/docs/exposure-design.md): LP formulation and simplex solver details.
- [`docs/templates.md`](file:///home/mango/Repos/muadib/docs/templates.md): Candidate 57-template rule set.

---

## 3. Output Hints & Checkpoint Chains

- Checkpoint paths: `outputs/pilot-run-20261101-01/pilot.pt`, `outputs/pilot-smoke-20261101-01/pilot.pt`
- Report files: `outputs/exposure-57-holdout-76001.json`, `outputs/design-samples-57.json`, `outputs/checker-perturb-41000.json`
- Render outputs: `outputs/example.png`, `outputs/example.json`, `outputs/conditioned.png`
