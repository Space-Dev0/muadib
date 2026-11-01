# Stage 0 status

**Not passed.** The following measurements are instrument-development results,
not confirmatory research findings.

| Component | Current evidence | Remaining work |
| --- | --- | --- |
| Renderer and rules | 2D renderer, annotation-based rule evaluation and grammar witnesses | Finalize and validate the benchmark vocabulary and full counterbalancing. |
| Candidate templates | Provisional 57-template generator covers all singleton colors and shapes in each one-attribute cell; a 100-scene-per-template sampling smoke test completed | Confirm full counterbalancing, held-out rules and sufficient sampling precision. |
| Pilot training | Exploratory online conditional diffusion instrument; CPU unit tests and a one-step train/checkpoint/sample CLI smoke passed in a separate CPU PyTorch 2.9.1 environment | No actual CUDA training, validated model outputs, reference model, resume, controlled guidance sweep, or human audit; GPU fit unverified. See [pilot training](pilot-training.md). |
| Checker | Clean-render and controlled perturbation audits; conservative sharpening fallback | High rejection remains at brightness 1.10, blur 0.50, and blur 1.00; validate every relevant rule and audit generated images with human judgments. |
| Exposure matrix and design | Monte Carlo estimates, LP feasibility and numerical minimum-KL allocation | Quantify matrix uncertainty and verify feasible exposure manipulations and leave-out probes on the final benchmark. |
| Decorrelation | Twenty exploratory hit-and-run designs from the 57-condition pilot; pooled correlations and VIF computed over 1,140 rule-by-design rows | Define breadth, freeze the analysis table and uncertainty method, and evaluate the registered criteria. Current estimates do not assess the proposal gate. |
| Gaussian diagnostic | One-dimensional analytic Bayes noise-loss reference and Monte Carlo check | Connect normalization checks to the eventual model/loss implementation. |
| MOSAIC contrast | Preliminary paper/code review | Run and document the adapted replication; see [MOSAIC notes](mosaic-replication.md). |
| Human audit | Blinded preparation, prediction and scoring for rules and independently judged predicate events | Obtain model-generated images and actual independent human judgments; none have been fabricated. |

For the current checker, a 200-scene holdout on separate synthetic seeds
41000–41199 yielded coverage 200/200 for clean and 0.90 brightness, 114/200 for
brightness 1.10 and blur radius 0.5, and 31/200 for blur radius 1.0. The
previous implementation baseline on these seeds was 85/200 for blur radius 0.5.
No rule errors were observed on accepted examples;
rejected scenes are excluded, so these counts do not establish overall checker
validity. These figures replace the earlier 100-scene results from an earlier
checker version. Reproduce with:

```sh
uv run python -m exposure.checker_perturb --samples 200 --seed 41000 --output outputs/checker-perturb-41000.json
```

The checker's sharpening fallback is attempted only when the original palette
validation passes. It accepts
only if two mild `UnsharpMask` settings with radii `0.5` and `0.6` and percent
`100` return identical object labels and exact centers. The fallback does not
validate generated images, pass Stage 0, or ensure robustness.

The proposal's checker, design-decorrelation and MOSAIC gates must be evaluated
using the agreed final instruments and data. No actual CUDA model training or
Stage 1 pilot has been run; only CPU pilot mechanics have been exercised. The
remote `uv` environment is being prepared and the CUDA PyTorch download is in
progress, so GPU fit is unverified, and repository transfer awaits explicit
authorization. No GPU-hours estimate has yet been measured.

A 300-scene clean-render audit at seeds 30000–30299 covered all 57 candidate
rules, with zero detector failures and zero observed rule errors. Some rules
had as few as five positive scenes, so these counts do not establish a 2% error
bound. A 100-scene-per-condition matrix smoke test at seed 43 yielded an
estimated singleton-preserving equality rank of 10 and nullity of 47. Twenty
designs were sampled with seed 43 and a 0.001 minimum weight. All 1,140
rule-by-design rows had finite marginal and lift estimates. The pooled
log-mean-marginal/log-lift correlation was -0.446; VIF was 11.61 for log lift
and 13.72 for relation count. These rows share rules and designs, and the
exposures and equality subspace come from the same small Monte Carlo matrix.
The statistics are exploratory, not confirmatory and do not establish the
proposal's decorrelation criteria. Reproduce with:

```sh
uv run python -m exposure.design_sample \
  --report outputs/exposure-57-pilot.json \
  --weights outputs/uniform-57.json \
  --samples 20 --seed 43 --min-weight 0.001 \
  --burn-in 100 --thin 10 --output outputs/design-samples-57.json
uv run python -m exposure.design_diagnostics \
  --report outputs/exposure-57-pilot.json \
  --designs outputs/design-samples-57.json \
  --output outputs/design-diagnostics-57.json
```

Local reports are written under `outputs/` and are not tracked in Git.
