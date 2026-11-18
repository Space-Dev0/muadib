# Research review: verification and actionable goals

This plan responds to the independent review received on 18 November 2026. The review is an input, not authority: check each claim against current source, artifacts and primary literature. The central joint-versus-marginal hypothesis has not been tested; Stage 0 is not passed. The factorial mixture design below is a **candidate**, not a demonstrated solution or a commitment to train models before feasibility and checker gates pass.

## Claims checked

1. **Contrast discrepancy — not established.** The frozen `paper/review-aggregate.json` and `docs/06-mosaic/mosaic-matched-eval-82000.md` agree on the following counts:

   | Cell | Yes | No | Unresolved |
   | --- | ---: | ---: | ---: |
   | Low / balanced | 16 | 0 | 0 |
   | Low / imbalanced | 15 | 0 | 1 |
   | High / balanced | 14 | 2 | 0 |
   | High / imbalanced | 14 | 1 | 1 |

   Total 59/3/2. Thus one unresolved answer occurs in low/imbalanced; the review's visual reconstruction with 16 yes there is incorrect. Put $a,b\in\{0,1\}$ for the unresolved low/imbalanced and high/imbalanced outcomes. Then $D_C=(3+a-b)/32$, $D_I=(1-a-b)/32$, and $D_C-D_I=(1+a)/16$. The extrema are respectively [0.0625, 0.125], [-0.03125, 0.03125], and [0.0625, 0.125]. These are finite-image bounds, not confidence intervals. The plotting source already sets both bar axes to start at zero. To avoid visual ambiguity, the paper now gives the table and numbers the nonzero stacked bar segments.

2. **Public repository — currently public, external access not fully verified.** `gh repo view` on 18 November reported `PUBLIC` and `isPrivate: false`. An unauthenticated browser fetch returned a cache failure, which is not evidence of either access or inaccessibility. The project remains [here](https://github.com/Space-Dev0/muadib); check from a logged-out connection separately. Public code does not imply that ignored review artifacts or checkpoints are publicly available.

3. **Object-count dependence of scene-level lift — valid counterexample, but not monotonic in general.** Under the review's iid, independent per-object attribute assumptions and fixed K, `m_A=1-(1-a)^K`, `m_B=1-(1-b)^K`, `n=1-(1-ab)^K`. With `a=b=0.5`, lift is 0 at K=1, approximately -0.2513 at K=2, -0.2809 at K=3, and -0.0560 at K=10; it approaches 0 as K grows. Therefore it is not mechanically *decreasing for all K*. The example does show a genuine event-definition and object-count dependence. The renderer's grid, color/shape sampling, conditioning and variable object counts need separate analysis; the iid formula is not automatically its null model. Object-level estimands or adjusted scene baselines require explicit role/binding semantics and validation, not automatic substitution.

4. **Pilot memorization — risk, not a measured outcome.** 500,000 updates × batch 4 / 200 scenes = 10,000 dataset-equivalent exposures. Sampling is stochastic, so this is not necessarily 10,000 complete sequential epochs, and exposure count alone does not prove memorization. A nearest-neighbor analysis with appropriate controls is still required.

5. **Fixed effects and optimization — substantive concern.** If a marginal covariate is constant within each rule, its coefficient is exactly collinear with unrestricted rule intercepts. The suggested four global mixtures may avoid this only if feasible mixtures generate sufficient independent within-rule variation; log determinant alone neither guarantees factorial low/high cells, rank, acceptable conditioning, population constraints, nor adequate uncertainty. Test these properties before considering training. Existing D-optimal allocation operates on fixed feature rows; it is not this proposed joint optimization.

6. **Checker and review provenance — supported.** Four returned CSVs in the 64-image study are byte-identical, so independence is not established by the artifacts. The experimental checker abstained on every human-negative and unresolved image. Recent segmentation-based rule-adapter evaluation also failed to establish a better validated checker; see `docs/01-instruments/image-diagnostics.md`. Review claims about a structural "always abstains when target absent" mechanism need code-level and negative-case testing, not inference from three samples.

7. **Guidance and citations — verified and corrected in the manuscript.** The existing sampler uses `unconditional + guidance * (conditional - unconditional)`, with guidance 1 selecting conditional prediction. The matched run specifies dropout 0.1, learning rate 1e-4 and EMA decay 0.999. The manuscript now states this convention and adds Salimans & Ho's [velocity-parameterization paper](https://arxiv.org/abs/2202.00512) and [classifier-free guidance](https://arxiv.org/abs/2207.12598). The dropout choice alone does not imply an inference-time improvement at guidance 1.

## Prioritized goals

### G1 — Make the current evidence auditable before the next presentation

- [x] Add the exact cell-count table and clarify unresolved-label placement in the paper.
- [x] Make small stacked bar segments numerically readable; retain zero-based axes.
- [x] Document guidance convention and cite the original relevant papers.
- [ ] Verify the public repository URL from a logged-out browser; inventory which code, images, reports and weights are unavailable to outside reviewers. Decide what can be shared without releasing private review material or host addresses.
- [ ] Add a self-contained methods appendix: renderer vocabulary and conditioning, rule inventory, code revision and artifact hashes, checker decisions/abstentions, model architecture and verified optimizer/sampler settings. Distinguish the 57-rule exposure instrument from the separate four-cell pilot.
- **Acceptance:** every numerical table/plot can be reproduced from a named artifact and checked without guessing an unresolved label; all important claims carry scope and denominators.

### G2 — Define the estimand and test a four-mixture design on CPU

- [ ] Specify a small fixed target-rule set and whether inference concerns within-rule joint exposure at fixed marginals or two varying effects. State the population, exposure definitions, model and contrasts before optimizing.
- [ ] Implement an exploratory global four-mixture optimizer using empirical matrices. Optimize the within-rule-centered information matrix subject to explicit simplex, minimum weights, marginal tolerances and predeclared low/high contrasts. Handle zero/near-zero exposures explicitly and constrain candidate distributions to be implementable. Include feasible baseline and multiple starts.
- [ ] On held-out renderer draws, report feasibility, achieved contrasts, per-rule coverage, weighted within-rule rank and singular values, condition number, determinant, marginal drift and uncertainty. Test null simulations that recover known effects and compare with existing endpoint and uniform baselines.
- **Acceptance:** a frozen candidate achieves declared low/high contrasts and full within-rule rank on independent measurement draws, with a stable conditioning margin and acceptable uncertainty. Failure to find one does not prove impossibility; do not claim four models suffice until constraints are verified.

### G3 — Repair the benchmark and distinguish object versus scene semantics

- [ ] Develop a binding-aware scene/event null; compare object-level and scene-level lift in simulations where object-count and attribute distributions are known. Include random K and realistic renderer dependencies, not just the iid fixed-K example.
- [ ] Use the existing factorial template candidate as a starting point, but address saturated relation events and verify sampling support, counterbalancing, complexity coverage, and independently generated train/holdout scenes. Do not reinterpret existing `left_of` semantics or historical labels silently.
- **Acceptance:** documented predicates with nondegenerate prevalence, supported rule conditions, held-out scene separation and a passing independent benchmark audit; revised renderer must use versioned semantics.

### G4 — Validate the checker and independent human audit

- [ ] Extend the [balanced synthetic calibration candidate](../04-checker-and-human-audit/synthetic-calibration-82800.md) with overlapping and genuinely ambiguous cases; freeze any candidate detector and define its decision/abstain logic. Separate synthetic and model-generated domains and train/calibration/test seeds.
- [ ] Obtain independently attributable blinded reviews on an untouched generated-image test set; record uncertainty and adjudication separately. Report a 2×3 truth-by-output table (yes/no/abstain), class-specific coverage, false accept/reject counts and uncertainty. Compute agreement only when independent attribution is supported. ROC requires a continuous calibrated score; do not require or report one for a hard-decision checker without such a score.
- **Acceptance:** predeclared per-class coverage/error targets and uncertainty criteria pass on the held-out set; a high selected-only accuracy or synthetic pixel IoU is insufficient.

### G5 — Resolve memorization and run an exposure intervention only after gates

- [ ] Extend the initial [pixel nearest-neighbor audit](../06-mosaic/memorization-nearest-neighbors.md) of frozen pilot outputs against their actual training data with calibrated transformed-copy/non-copy controls and robust duplicate/near-duplicate checks. Report distances and visual examples; pixel L2 alone is not a complete memorization test.
- [ ] Extend and validate the [counterbalanced, geometry-paired scene candidate](../06-mosaic/counterbalanced-pilot-data.md), including systematic distractor-dependence checks, plus independent dataset/model seed blocks. Freeze splits, mixtures, evaluation and stopping rules; run a bounded smoke test before a fully budgeted multi-seed experiment.
- [ ] Train multi-predicate conditioned models only after G2–G4, then report all allocated comparisons including failures and uncertainty. Five seeds, 500 calibration examples and larger datasets in the review are proposals, not universal guarantees of statistical power; calculate resource and power needs from the chosen estimand.
- **Acceptance:** predefined held-out outcomes, matched interventions, independent model/dataset replicates and an analysis that estimates the stated contrast. No endpoint-only design, image-level seed count, or checker abstention can substitute for model replication.

## Suggested ownership

One person can lead G1 and artifact/repository checks; one G2 statistical design and simulations; one G3 renderer and benchmark; one G4 human/checker protocol. All four review G5's frozen protocol and compute budget before training. This is a division of tasks, not evidence that reviewers or compute resources are already available.
