# Project Analysis Summary: Which Exposure Matters? (`muadib`)

## 1. Executive & Repository Overview
`muadib` implements the experimental framework for the research proposal **"Which Exposure Matters? Controlled experiments on exposure in conditional diffusion models" (Rev. 4)**.
The project investigates whether the compositionality cost in conditional generative models (specifically conditional diffusion / flow matching) is governed by joint rule exposure during training versus individual constituent predicate exposures, while holding marginals constant.

- **Target Repository:** `/home/mango/Repos/muadib`
- **Language / Runtime:** Python 3.11+, PyTorch 2.9.1+cu128 (optional extra via `--extra train`), Pillow, NumPy, SciPy.
- **Test Suite Status:** 209 tests discovered; 154 passing on CPU, 55 skipped (which require the optional PyTorch training extra).
- **Research Stage:** Instrument Development (Stage 0). Stage 0 gate is explicitly marked **Not passed** in current documentation.

---

## 2. Structural & Architectural Layout

The codebase is organized into four main layers:

### A. Core Semantics & Synthetic Rendering
- [`exposure/rules.py`](file:///home/mango/Repos/muadib/exposure/rules.py): Formal semantics for scene predicates (`SceneObject`, `Attribute`, `Relation`). Evaluates rule conjunctions over distinct objects using existential quantification and permutations.
- [`exposure/render.py`](file:///home/mango/Repos/muadib/exposure/render.py): 2D synthetic scene generator. Renders 64×64 RGB scenes of 3 objects on a 3×3 grid using PIL, with deterministic seeds and rejection sampling for conditional generation.
- [`exposure/grammar_audit.py`](file:///home/mango/Repos/muadib/exposure/grammar_audit.py): Audits canonical rule layouts and generates witnesses.
- [`exposure/templates.py`](file:///home/mango/Repos/muadib/exposure/templates.py) & [`exposure/template_balance.py`](file:///home/mango/Repos/muadib/exposure/template_balance.py): 57-rule candidate template generator and per-cell vocabulary coverage checks.

### B. Exposure Measurement, Design & Feasibility
- [`exposure/measure.py`](file:///home/mango/Repos/muadib/exposure/measure.py): Computes Monte Carlo plug-in co-satisfaction matrices ($A$-matrices) and Wilson 95% confidence intervals across rules and predicate subsets.
- [`exposure/mixture.py`](file:///home/mango/Repos/muadib/exposure/mixture.py): Validates mixture reports and computes weighted exposures and log lift ($L_r = \log n_r - \sum \log m_p$).
- [`exposure/design.py`](file:///home/mango/Repos/muadib/exposure/design.py): Solves simplex LP feasibility for rule/marginal constraints and performs minimum-KL allocation optimization with SciPy SLSQP.
- [`exposure/design_subspace.py`](file:///home/mango/Repos/muadib/exposure/design_subspace.py), [`exposure/design_sample.py`](file:///home/mango/Repos/muadib/exposure/design_sample.py), [`exposure/design_strata.py`](file:///home/mango/Repos/muadib/exposure/design_strata.py), [`exposure/design_diagnostics.py`](file:///home/mango/Repos/muadib/exposure/design_diagnostics.py): SVD analysis of singleton-preserving equality directions, hit-and-run sampling of mixture weights, and diagnostic VIF/correlation checks.
- [`exposure/validate_mixture.py`](file:///home/mango/Repos/muadib/exposure/validate_mixture.py): Validates predicted vs empirical scene distributions.

### C. Generative Modeling & Diffusion Instruments
- [`exposure/pilot_train.py`](file:///home/mango/Repos/muadib/exposure/pilot_train.py): Online conditional diffusion instrument (`PilotUNet`). Renders fresh scenes on each step, tokenizes predicate slots (subject, object, attribute), predicts DDPM noise with cross-attention at 16×16, and supports classifier-free guidance, EMA, and deterministic CUDA execution (v3 checkpoints).
- [`exposure/velocity_online.py`](file:///home/mango/Repos/muadib/exposure/velocity_online.py) & [`exposure/velocity_guidance.py`](file:///home/mango/Repos/muadib/exposure/velocity_guidance.py): Online velocity-matching / flow-matching formulation predicting velocity $v = x_1 - x_0$.
- Diagnostic probes:
  - [`exposure/denoise_probe.py`](file:///home/mango/Repos/muadib/exposure/denoise_probe.py): One-step denoising MSE probe.
  - [`exposure/reconstruction_probe.py`](file:///home/mango/Repos/muadib/exposure/reconstruction_probe.py): Truncated ODE/SDE reconstruction accuracy.
  - [`exposure/high_t_baseline.py`](file:///home/mango/Repos/muadib/exposure/high_t_baseline.py) & [`exposure/high_t_finetune.py`](file:///home/mango/Repos/muadib/exposure/high_t_finetune.py): High-$t$ loss isolation and fine-tuning.
  - [`exposure/identity_eps_finetune.py`](file:///home/mango/Repos/muadib/exposure/identity_eps_finetune.py): Constant-identity diagnostic.
  - [`exposure/time_embedding_probe.py`](file:///home/mango/Repos/muadib/exposure/time_embedding_probe.py): Time-embedding capacity probe.
  - [`exposure/optimizer_probe.py`](file:///home/mango/Repos/muadib/exposure/optimizer_probe.py): AdamW vs SGD optimizer performance comparison on fixed targets.
  - [`exposure/gaussian_diagnostic.py`](file:///home/mango/Repos/muadib/exposure/gaussian_diagnostic.py): 1D Gaussian mixture analytic Bayes denoiser reference.

### D. Validation & Auditing
- [`exposure/checker.py`](file:///home/mango/Repos/muadib/exposure/checker.py): Heuristic pixel-based checker parsing 64×64 scenes back to `SceneObject`s.
- [`exposure/checker_perturb.py`](file:///home/mango/Repos/muadib/exposure/checker_perturb.py): Evaluates checker accuracy under brightness and blur perturbations.
- [`exposure/human_audit.py`](file:///home/mango/Repos/muadib/exposure/human_audit.py) & [`exposure/audit_agreement.py`](file:///home/mango/Repos/muadib/exposure/audit_agreement.py): Blinded human evaluation schema and descriptive pairwise agreement.

---

## 3. Entrypoint Summary

| Task | Module / CLI Command | Primary Artifacts / Outputs |
| :--- | :--- | :--- |
| **Synthetic Render** | `python -m exposure.render --seed <int> [--rule <json>] --output <stem>` | `<stem>.png`, `<stem>.json` |
| **Exposure Measure** | `python -m exposure.measure --rules <json> --samples <int> --output <json>` | Report JSON ($A$-matrix, Wilson intervals) |
| **Design LP Solver** | `python -m exposure.design --report <json> --spec <json> --output <json>` | Allocation weights $w$ JSON |
| **Diffusion Training** | `python -m exposure.pilot_train train --rules <json> --weights <json> --output <ckpt.pt>` | Schema v3 PyTorch checkpoint |
| **Diffusion Sampling** | `python -m exposure.pilot_train sample --checkpoint <pt> --rule <name> --output <png>` | Generated sample PNG |
| **Velocity Training** | `python -m exposure.velocity_online train --rules <json> ... --output <ckpt.pt>` | Schema v1 Velocity checkpoint |
| **Pixel Checker** | `python -m exposure.checker --rules <json> --image <png> [--json]` | Detected objects & satisfied rules |
| **Unit Test Suite** | `uv run python -m unittest discover -s tests -v` | Test results (209 test cases) |

---

## 4. Configuration & Specification Files

- **Rule Definitions (`examples/exposure_rules.json`, `examples/red_left_of_square.json`):** Ordered mapping of rule names to conjunctions of attribute and relation predicates.
- **LP Design Constraints (`examples/exposure_constraints.json`):** Lower/upper bounds for full rules, predicate marginals, and pairs.
- **Allocation Weights (`examples/uniform_weights.json`):** Normalized categorical distribution over conditioning rules.

---

## 5. Conservative Suggestions & Insertion Points

1. **Estimand Metric Calibration (`exposure/composition.py`, `exposure/mixture.py`):**
   - Both `Γ^log` and `Γ^logit` are exactly zero under `RSR = Π c_p`; they are
     different transforms, not competing nulls. `Γ^log` is now the documented
     primary and `Γ^logit` the sensitivity. The original review's
     `logit(Π c_p) − Σ logit(c_p)` expression is a different quantity and is not
     implemented. Boundary values are reported as `null` with a status.
2. **Model Architecture Insertion Point (`exposure/pilot_train.py` & `exposure/velocity_online.py`):**
   - `PilotUNet` currently uses cross-attention over predicate tokens without spatial 2D coordinate embeddings. When upgrading the backbone (e.g. DiT or larger UNet), insert spatial positional encodings in `Residual` / attention blocks.
3. **Checker Modernization (`exposure/checker.py`):**
   - The current pixel checker functions as an inverted clean renderer with rigid bounding box and component count assumptions. For model-generated images with texture/noise, integrate a pre-trained segmentation/detection model or calibrated classifier-based detector.
4. **Counterbalanced Rule Generation (`exposure/templates.py`):**
   - Expand the 57-template set into a fully counterbalanced factorial grammar to prevent confounding between attribute types (colors/shapes) and relation categories (`left_of` vs `above`).
