# Human checklist for Stage 0

**Stage 0 has not passed.** This checklist lists work that requires the human
researcher; it is not evidence that any gate is met.

This checklist is derived from the proposal
`Which Exposure Matters Research Proposal (Rev. 4).pdf` and the current
repository documentation, especially [Stage 0 status](stage0-status.md),
[research and hardware plan](research-and-hardware-plan.md),
[experiment specification](experiment-spec.md),
[MOSAIC replication notes](mosaic-replication.md), and the
[human audit workflow](human-audit.md). The proposal records the Stage 0
go/no-go criteria in Sections 5, 6, and 12 (pp. 4-5, 5, and 8-9); this checklist
quotes those criteria exactly and does not add thresholds. This document may
change as instruments and data are finalized, but the proposal criteria below
remain the reference for sign-off.

Labels used below:

- **Human decision/action** — cannot be completed by the current code.
- **Code already supports** — existing tooling or evidence. This is
  instrumentation, not independent validation or a gate pass.

## Proposal Stage 0 go/no-go criteria

The proposal's Stage 0 go condition (Section 12, pp. 8-9) requires all three of
the following:

- [ ] **Checker error ≤ 2% on real renders** (Section 5, p. 5; Section 12,
  p. 9).
- [ ] **Lift decorrelated from marginal exposure**, with absolute correlation
  `|r| ≤ 0.3` and `VIF < 2` (Section 6, p. 5; Section 12, p. 9).
- [ ] **The small adapted MOSAIC replication matches MOSAIC's direction**
  (Section 12, pp. 8-9).

The checker evidence is specified as a human-audited, complexity-stratified
sample of at least 300 generated images from a pilot model, with false-accept
and false-reject rates reported per predicate and per rule (Section 5, p. 5).
The `2%` target applies on real renders.

**None of these criteria is claimed met here.** The current repository has
instrument-development evidence only.

## 1. Decisions to make and record

- [ ] **Human decision/action:** Finalize and lock the benchmark vocabulary.
  The proposal (Section 5, p. 4) specifies 64×64 images, three objects, 3 shapes
  × 4 colours, texture only if a pilot shows it is legible at this scale, and
  `left_of`, `above`, plus one further relation after the pilot. Decide whether
  texture or the third relation is retained, dropped, or moved to 96×96, and
  record the final grammar and semantics.
  **Code already supports:** the current 2D renderer, four colors, three
  shapes, two strict relations, three-object scenes, and annotation-based rule
  evaluation. No texture or third relation is implemented.
- [ ] **Human decision/action:** Lock pair and marginal exposure semantics in the
  experiment specification before analysis, including distinct-object binding
  and existential re-binding within each subset.
  **Code already supports:** distinct assignments, shared bindings within a
  rule or subset, and subset-wise existential evaluation.
- [ ] **Human decision/action:** Define the final template set against the
  proposal's "about 54 training templates across the feasible cells, plus a
  held-out set for the leave-out probes (H3)", with concrete shapes, colours,
  and relations counterbalanced across complexity levels (Section 5, p. 4).
  Decide whether the provisional 57-template set (seven candidates in each
  `K_attr = 1` cell, six elsewhere) is final, needs further counterbalancing, or
  needs held-out rules.
  **Code already supports:** a deterministic 57-template generator with
  validated layouts, canonical duplicate checks, witnesses, and manifests. It
  does not establish full counterbalancing or provide the proposal's held-out
  leave-out set.
- [ ] **Human decision/action:** Choose the sampling precision and acceptance
  budget for every rule. The proposal estimates each A-matrix entry once with
  about `10^5` scenes per template (Section 6, p. 5); decide the final Monte
  Carlo size and confirm the bounded rejection sampler reaches each final rule
  often enough. Treat sampling failure as unresolved, not as proof of
  impossibility.
  **Code already supports:** bounded rejection sampling, attempt counts, and a
  100-scene-per-template smoke test. It does not run the proposal's
  approximately `10^5` scenes per template.
- [ ] **Human decision/action:** Quantify matrix uncertainty and decide how it
  will be propagated (or explicitly not propagated) into feasibility, lift, and
  design results. Freeze the analysis table and uncertainty method.
  **Code already supports:** Monte Carlo estimates, pointwise Wilson intervals,
  LP feasibility, minimum-KL allocation, and exploratory correlation/VIF
  diagnostics. It does not propagate matrix uncertainty.
- [ ] **Human decision/action:** Define breadth's partner-event distribution,
  including how roles and multiple co-occurrences count, before using breadth as
  a covariate.
  **Code already supports:** no breadth implementation.
- [ ] **Human decision/action:** Define zero-aware statistical models and
  probability/logit handling. Distinguish exposure probability `n50` from total
  exposure count `N50` at a fixed budget.
  **Code already supports:** explicit zero statuses and null log lifts. No
  zero-aware model or `N50` analysis is implemented.
- [ ] **Human decision/action:** Evaluate the proposal's decorrelation criterion:
  lift must be decorrelated from marginal exposure with absolute correlation
  `|r| ≤ 0.3` and `VIF < 2` (Section 6, p. 5; Section 12, p. 9). Freeze the
  analysis table, define breadth, and choose the uncertainty method before
  treating any result as the gate evaluation.
  **Code already supports:** exploratory hit-and-run designs, pooled
  correlations, and VIF reports over the 57-condition pilot. The current
  exploratory point estimates are a pooled log-mean-marginal/log-lift
  correlation of `-0.446` and VIFs of `11.61` for log lift and `13.72` for
  relation count over 1,140 rule-by-design rows. These are **not** the
  registered gate analysis: rows share rules and designs, breadth is omitted,
  uncertainty is not propagated, and the exposures and equality subspace come
  from the same small Monte Carlo matrix. The values do not establish the
  proposal criterion, and they are not a formal pass/fail decision.
- [ ] **Human decision/action:** Decide how the checker's high rejection under
  brightening and blur will be improved or explicitly handled so the proposal's
  checker error target of `≤ 2%` on real renders can be evaluated (Section 5,
  p. 5; Section 12, p. 9). Rejection rates must be reported alongside error
  rates; do not treat low errors among accepted images as overall validity.
  **Code already supports:** clean and controlled-perturbation audits with
  rejection counts reported separately. It does not establish the `≤ 2%`
  target on real or generated renders.
- [ ] **Human decision/action:** Decide how the one-dimensional Gaussian
  diagnostic maps onto the eventual model and loss implementation, including
  normalization conventions.
  **Code already supports:** an analytic Bayes noise-loss reference and a Monte
  Carlo check for a one-dimensional Gaussian mixture.
- [ ] **Human decision/action:** Decide whether the finite synthetic dataset or
  an external dataset is the secondary experiment. If external, assess
  annotations, license, condition labels, and feasibility for controlled
  exposures before acquisition.
  **Code already supports:** neither an external dataset nor a finite offline
  dataset pipeline. The primary plan is online scene generation.
- [ ] **Human decision/action:** Lock pilot choices, exclusions, and the final
  analysis plan before confirmatory runs. Treat the current experiment
  specification as a working draft, not a preregistration.
  **Code already supports:** reproducible commands and seeded reports. No
  preregistration or confirmatory run exists.

## 2. Access and data acquisition

- [ ] **Human decision/action:** Obtain or generate the model-generated images
  needed for the human image audit. The current project has no trained diffusion
  model and no generated-image set.
  **Code already supports:** blinded manifest preparation, checker prediction,
  and scoring once images and judgments exist.
- [ ] **Human decision/action:** Provide the proposal's human-audited,
  complexity-stratified sample of at least 300 generated images from a pilot
  model, with enough images in every stratum the review manifest requests.
  Confirm the final sample size, strata, and sampling seed (Section 5, p. 5).
  **Code already supports:** stratified sampling from each present stratum and
  failure when a present stratum is too small. Missing strata are not invented.
- [ ] **Human decision/action:** Record that
  `Which Exposure Matters Research Proposal (Rev. 4).pdf` is the authoritative
  gate definition used for sign-off. If a newer revision appears, reconcile it
  with Sections 5, 6, and 12 before continuing.
  **Code already supports:** no automatic gate-version registry.
- [ ] **Human decision/action:** If an external dataset is selected, secure
  access, verify the license and annotation quality, and confirm that condition
  labels support the planned exposures.
  **Code already supports:** no dataset download, loader, or license audit.
- [ ] **Human decision/action:** Arrange independent human reviewer access,
  including a neutral delivery method for images and manifest that does not
  reveal checker predictions or gold labels.
  **Code already supports:** blinding by omission and neutral-ID preparation.
  The tool cannot determine whether a user-chosen name itself leaks information.
- [ ] **Human decision/action:** Provide the MOSAIC paper, implementation, data,
  and any task-specific pretrained classifier weights or confirm lawful access
  and versions.
  **Code already supports:** only a preliminary source review; no MOSAIC data or
  code is bundled.

## 3. Human image review

- [ ] **Human decision/action:** Prepare the review manifest and images from the
  actual generated-image set, with one consistent predicate list per rule name
  and matching complexity counts.
  **Code already supports:** `exposure.human_audit prepare`, path containment
  checks, unique-ID checks, and protection against overwriting inputs.
- [ ] **Human decision/action:** Run the checker prediction step separately and
  keep its output away from reviewers.
  **Code already supports:** `exposure.human_audit predict`, producing
  `true`/`false`/`null` per sample without modifying the manifest.
- [ ] **Human decision/action:** Have independent humans judge whether each
  image satisfies its requested rule, using the exact `sample_id,judgment` CSV
  schema with `yes` or `no`.
  **Code already supports:** validation of the review CSV against manifest IDs,
  including duplicate and malformed-value rejection.
- [ ] **Human decision/action:** Score the completed reviews and report
  false-accept and false-reject rates **per predicate and per rule**, as the
  proposal requires (Section 5, p. 5), together with complexity-stratified and
  overall counts and rejection counts.
  **Code already supports:** `exposure.human_audit score`, conditional
  false-accept/false-reject rates, and explicit `null` for missing denominators.
  It reports overall, per-rule, and per-complexity results; it does not produce a
  separate per-predicate report, so that tabulation or tooling remains to be
  added.
- [ ] **Human decision/action:** Decide whether the observed human-checker
  agreement, per-predicate and per-rule errors, and rejection counts meet the
  proposal's `≤ 2%` checker-error target on real renders (Section 5, p. 5;
  Section 12, p. 9). Clean-render counts alone do not constitute independent
  certification.
  **Code already supports:** descriptive counts and rates. It makes no pass/fail
  decision and does not establish the `≤ 2%` target.

## 4. GPU and MOSAIC work

- [ ] **Human decision/action:** Identify the training machine and confirm its
  GPU model, driver, operating system, and Python version.
  **Code already supports:** CPU-only rendering and checks. The current
  environment did not expose `nvidia-smi`; this must be checked on the training
  machine.
- [ ] **Human decision/action:** Select and install the correct PyTorch CUDA
  build, then lock and verify the dependency set.
  **Code already supports:** no CUDA acceleration and no locked training
  dependency set.
- [ ] **Human decision/action:** Set a feasible GPU-hour budget, including
  expected failed runs, and record how it was estimated. The proposal's working
  assumption is about 3 GPU-hours per run on one modern GPU for roughly 60 runs,
  plus about 50 GPU-hours for diagnostics, on the order of **200-250
  GPU-hours**, to be replaced by pilot measurements, with a hard cap set after
  the pilot (Section 12, p. 9).
  **Code already supports:** no measured GPU-hours estimate and no training run.
  The 200-250 figure is an initial working assumption, not a measured result.
- [ ] **Human decision/action:** Read the full MOSAIC paper and relevant
  implementation, then specify the contrast, metrics, dataset sizes, and
  balancing conditions to reproduce.
  **Code already supports:** [preliminary notes](mosaic-replication.md) only.
- [ ] **Human decision/action:** Decide and document how this project's 2D
  adaptation differs from MOSAIC in task, complexity manipulation, conditioning,
  training budget, and checker.
  **Code already supports:** a different 2D renderer with three objects and an
  experimental image-based checker. No adapted contrast has been run.
- [ ] **Human decision/action:** Run and record the replication with seeds,
  configurations, code versions, uncertainty, and failed runs. Compare only the
  direction with MOSAIC unless a full reproduction is actually achieved.
  **Code already supports:** nothing for this gate beyond source review. CPU-only
  instrument checks do not count as the requested replication.
- [ ] **Human decision/action:** Defer novelty claims until the MOSAIC contrast
  has been read, reproduced or adapted, and documented.
  **Code already supports:** no novelty evidence.

## 5. Final Stage 0 gate sign-off

- [ ] **Human decision/action:** Confirm that every instrument and dataset used
  for the gates is the agreed final version, with code version, dependency lock,
  seeds, and any deviations recorded.
  **Code already supports:** reproducible commands for current instruments.
  Their outputs are not final gate evidence.
- [ ] **Human decision/action:** Evaluate all three proposal Stage 0 go/no-go
  criteria using the final instruments and data: checker error `≤ 2%` on real
  renders, including the human-audited complexity-stratified sample of at least
  300 generated images with per-predicate and per-rule false-accept/false-reject
  reporting; lift decorrelated from marginal exposure with `|r| ≤ 0.3` and
  `VIF < 2`; and the small adapted MOSAIC replication matching MOSAIC's
  direction (Sections 5, 6, and 12; pp. 4-5, 5, and 8-9).
  **Code already supports:** exploratory diagnostics only. Their own docs state
  they do not establish the gates, and no criterion is currently claimed met.
- [ ] **Human decision/action:** Confirm that all human judgments and generated
  images are real and independently produced, with no fabricated labels or
  images.
  **Code already supports:** no human labels or generated images.
- [ ] **Human decision/action:** Review the [Stage 0 status](stage0-status.md)
  table item by item and update it only with evidence that the agreed criteria
  are met.
  **Code already supports:** the current table, which records **Not passed**.
- [ ] **Human decision/action:** Record the final go/no-go decision in writing,
  including date, responsible person, evidence links, thresholds used, known
  limitations, and any approved exceptions.
  **Code already supports:** nothing. No command or artifact in this repository
  can sign off Stage 0.

## Minimum sign-off statement

A Stage 0 pass can be claimed only after the human has checked every applicable
box above, shown that all three proposal criteria are met using the final
instruments and data, and recorded an explicit go decision. Passing tests,
generating reports, or completing exploratory diagnostics do not by themselves
change the current **Not passed** status.
