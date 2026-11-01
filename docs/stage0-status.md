# Stage 0 status

**Not passed.** The following measurements are instrument-development results,
not confirmatory research findings.

| Component | Current evidence | Remaining work |
| --- | --- | --- |
| Renderer and rules | 2D renderer, annotation-based rule evaluation and grammar witnesses | Finalize and validate the benchmark vocabulary and full counterbalancing. |
| Candidate templates | Provisional 57-template generator covers all singleton colors and shapes in each one-attribute cell; a 100-scene-per-template sampling smoke test completed | Confirm full counterbalancing, held-out rules and sufficient sampling precision. |
| Checker | Clean-render and controlled perturbation audits | Improve or explicitly handle high rejection under brightening and blur; validate every relevant rule and audit generated images with human judgments. |
| Exposure matrix and design | Monte Carlo estimates, LP feasibility and numerical minimum-KL allocation | Quantify matrix uncertainty and verify feasible exposure manipulations and leave-out probes on the final benchmark. |
| Decorrelation | Twenty exploratory hit-and-run designs from the 57-condition pilot; pooled correlations and VIF computed over 1,140 rule-by-design rows | Define breadth, freeze the analysis table and uncertainty method, and evaluate the registered criteria. Current estimates do not assess the proposal gate. |
| Gaussian diagnostic | One-dimensional analytic Bayes noise-loss reference and Monte Carlo check | Connect normalization checks to the eventual model/loss implementation. |
| MOSAIC contrast | Preliminary paper/code review | Run and document the adapted replication; see [MOSAIC notes](mosaic-replication.md). |
| Human audit | Blinded preparation, prediction and scoring for rules and independently judged predicate events | Obtain model-generated images and actual independent human judgments; none have been fabricated. |

For the current checker, a 100-scene holdout at seeds 20000–20099 yielded
coverage 100/100 for clean and 0.90 brightness, 52/100 for brightness 1.10,
36/100 for blur radius 0.5, and 10/100 for blur radius 1.0. Rejections are
reported separately; error rates among accepted images cannot establish overall
checker validity. Reproduce with:

```sh
uv run python -m exposure.checker_perturb --samples 100 --seed 20000 --output outputs/checker-perturb-holdout.json
```

The proposal's checker, design-decorrelation and MOSAIC gates must be evaluated
using the agreed final instruments and data. No training or Stage 1 pilot has
been run, and no GPU-hours estimate has yet been measured.

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
