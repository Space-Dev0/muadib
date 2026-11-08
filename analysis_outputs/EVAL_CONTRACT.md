# Evaluation Contract: Which Exposure Matters?

- **Task Family:** `conditional_diffusion_exposure`
- **Dataset:** `synthetic_three_object_scenes (64x64 RGB)` generated on a 3×3 grid with bounded jitter.
- **Benchmark:** `Which Exposure Matters Research Proposal (Rev. 4)`
- **Primary Metrics:**
  - **Rule Satisfaction Rate (RSR):** Fraction of generated images evaluated as satisfying the requested rule condition $r$. Goal: `maximize`.
  - **Composition Gap ($\Gamma_r$):** Discrepancy between full rule performance and constituent performance.
    - Both scales are zero under the multiplicative null $RSR_r = \prod_p c_p$:
      $\Gamma_r^{\text{logit}} = \text{logit}(\text{RSR}_r) - \text{logit}(\prod_{p \in P_r} c_p)$
      and $\Gamma_r^{\log} = \log \text{RSR}_r - \sum_{p \in P_r} \log c_p$.
    - Pre-registered primary: $\Gamma_r^{\log}$ (additive over predicates). $\Gamma_r^{\text{logit}}$ is a sensitivity. The expression $\text{logit}(\prod c_p) - \sum \text{logit}(c_p)$ is a different quantity and must not be reported as $\Gamma$.
  - **Sample Complexity ($N_{50}$):** Number of training exposures required to achieve $\text{RSR} = 0.50$.
  - **Wilson 95% Confidence Intervals:** Pointwise binomial confidence bounds for Monte Carlo measurements.

---

## 1. Evaluation Commands

### Unit Testing & Regression Suite
```sh
uv run python -m unittest discover -s tests -v
```
- **Scope:** 209 test cases verifying rules, rendering, measurement, LP design feasibility, exact bounds, chain diagnostics, benchmark validity, human audit plumbing, and probes.
- **Target:** 154 tests pass unconditionally without GPU; 55 pass with `--extra train` installed.

### Rendering & Conditioning Verification
```sh
uv run python -m exposure.render --seed 42 --rule examples/red_left_of_square.json --output outputs/example
```
- Produces `outputs/example.png` and `outputs/example.json` containing ground-truth coordinates.

### Exposure Measurement
```sh
uv run python -m exposure.measure --rules examples/exposure_rules.json --samples 100 --include-subsets --output outputs/exposure-subsets.json
```
- Generates schema-1 co-satisfaction matrix including singleton and pair marginals.

### Image-Based Checker Verification
```sh
uv run python -m exposure.checker --rules examples/exposure_rules.json --image outputs/example.png --json
```
- Validates pixel-based rule detector outputs on synthetic scene renders.

---

## 2. Expected Artifacts

- Exposure reports: `outputs/exposure.json`, `outputs/exposure-subsets.json`
- Design solutions: `outputs/design-min-kl.json`, `outputs/design-samples-57.json`
- Checker perturbation reports: `outputs/checker-perturb-41000.json`
- Trained checkpoints: `outputs/pilot-run-*/pilot.pt` (Schema version 3)
- Sampling outputs: `outputs/pilot-run-*/*.png`

---

## 3. Stage 0 Go / No-Go Gate Criteria

Before Stage 1 confirmatory runs may begin, the following Stage 0 requirements must be met. Items 1–4 are **proposed** criteria; the repository's current Stage 0 documentation does not enforce a model criterion.

1. **Model Learnability (proposed):** Reference model reaches median reference RSR in $[0.3, 0.7]$ across test rules, and passes an overfit-1/10/100-scene pre-gate. *(Currently not met; 12k samples were noise, 32k produced two inspected samples).*
2. **Checker Calibration:** Pixel checker achieves false-positive rate $\le 0.05$ and false-negative rate $\le 0.10$ on model-generated images, audited by blinded human review. *(Currently evaluated only on clean/perturbed synthetic renders; perturbation rejection is 84.5–90% at blur radius 1.0 and accepted-image errors are zero, so the observed failure is attrition).*
3. **Design Decorrelation:** Final benchmark demonstrates $|r(\log \text{lift}, \log \text{marginal})| \le 0.30$ and $\text{VIF} < 2.0$ on an identified within-rule quantity. *(Current pilot pooled $r=-0.446$, VIF $11.61$; $\beta_m$ is exactly collinear with the rule intercept, so the pooled gate is not identified).*
4. **Benchmark Validity:** Relation singletons below 0.9, relation kind counterbalanced against endpoint binding, all 12 same-object color–shape pairs covered, and a held-out template split. *(Current `benchmark_audit` reports `confirmatory_ready: false`).*
5. **Human Review Agreement:** Human-checker agreement achieves Fleiss' $\kappa \ge 0.70$.
