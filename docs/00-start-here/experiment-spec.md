# Experiment specification: working draft

This document records prototype semantics. It is not a preregistration: the
benchmark, checker, exposure design, and analysis choices must be validated and
locked before confirmatory runs.

## Events and object bindings

A scene contains three annotated objects. A rule is a nonempty conjunction of
attribute and relation predicates over named variables. A scene satisfies a
rule when some assignment of **distinct** scene objects to distinct variables
satisfies all predicates simultaneously. Repeated variable names use the same
object. Coordinates increase rightward and downward; `left_of` and `above` use
strict inequalities on bounding-box centers. Ties fail the corresponding
relation.

For predicate or pair exposure, evaluate that subset as its own existential
rule, retaining variable names and shared bindings **within** the subset.
Different subsets may use different assignments. Two singleton events can both
hold even when their conjunction fails because distinct variables need distinct
objects. Therefore the product of singleton rates is a heuristic reference,
not necessarily the joint probability of those singleton events.

The current prototype has plain circles, squares and triangles; four colors;
and `left_of` and `above`. It does not yet support texture, a third relation,
the full proposed template set, or validated image-based measurements of model output.

## Exposure and conditioning

For each conditioning template `c`, the matrix column estimates the fraction
of freshly sampled annotated scenes satisfying each rule or predicate subset.
For mixture weights `w`, scene exposure is the weighted sum across columns.
Incidental satisfaction counts even when the scene was requested for a
different template. A full-rule success implies its own subset successes, but
unrelated rule events may also hold.

An explicitly presented condition and an incidentally satisfied condition are
different training signals. The current measurement pipeline computes the
scene-event matrix; it does not yet implement training or log condition-input
exposures. Future training must record both the sampled template and the
realized scene annotations, and must report these exposure types separately.

## Composition gap

For a rule `r` with singleton marginals `c_p`, define both scales:

```
Γ^logit = logit(RSR_r) − logit(Π_{p∈P_r} c_p)
Γ^log   = log RSR_r − Σ_{p∈P_r} log c_p
```

Both are exactly zero under the multiplicative null `RSR_r = Π c_p`; they are
different transforms, not competing nulls. `Γ^log` is the primary estimand
because it is additive over predicates. `Γ^logit` is reported as a sensitivity.
A saturated singleton (`c_p = 1`) makes `logit` undefined, so boundary values
are reported with an explicit status rather than dropped. The product null is
still a heuristic when predicates share object variables; a binding-aware
maximum-entropy null is the intended refinement and is not yet implemented.
See `exposure/composition.py`.

## Lift, zeroes, and uncertainty

Rule and pair log lift divide the estimated scene joint exposure by the product
of their respective singleton rates. The reference depends on the binding
semantics above. No independence or causal claim follows from a lift value.
The mixture report marks zero empirical marginals or joint counts with explicit
statuses and a null log lift. In particular, a zero observed joint is not proof
of zero probability; it does not establish a successful leave-out design.

Matrix cells are Monte Carlo estimates. Pointwise Wilson intervals do not give
simultaneous coverage across cells; measurements sharing scenes are correlated.
Mixture lift and LP/SLSQP designs do not yet propagate this uncertainty. Exact
zero-exposure claims require a separate design and uncertainty analysis.

## Pending decisions and gates

- **Estimand lock.** Pre-register `Γ^log` as primary and `Γ^logit` as
  sensitivity, including how boundary (`0`, `1`) cases are reported. Verify
  both are zero under a simulated multiplicative null. Do not use the
  retracted review expression `logit(Π c_p) − Σ logit(c_p)`.
- **Relation binding.** Bind both endpoints of every relation to attributed
  objects, counterbalance relation kind within complexity cells, cover all `12`
  same-object color–shape pairs, and target relation singleton exposure in
  `[0.2, 0.8]`. The current existential relations have analytic exposure
  `0.999271` and cannot support a relational contrast.
- **Learnability gate.** Add a pre-gate: overfit one fixed scene, then 10, then
  100 scenes, then the full template mixture; require median reference RSR in
  `[0.3, 0.7]` with a CI excluding zero. If it fails, report the failure as the
  result and stop. Stage 0 currently has no model criterion.
- **Coefficient-targeted design.** Replace "maximize closeness to uniform" with
  a D-optimal or c-optimal design for `β_λ` and `β_λR`, or directly maximize
  the minimum within-rule lift range. Report exact LP bounds and multi-chain
  R-hat/ESS alongside every sample; the current sampler realizes only `0.0955`
  log units of a `1.172`-wide polytope.
- **Checker validation.** Validate on generated images, report per-rule
  rejection and conditional/unconditional sensitivity/specificity, and use a
  latent-class or measurement-error correction. Current perturbation errors are
  attrition: `84.5–90%` rejection at blur radius `1.0`.
- **Template holdout.** Replace the hand-designed bundle set with a factorial,
  counterbalanced design and add a stratified held-out template split.
  `exposure/benchmark_audit.py` currently reports `confirmatory_ready: false`.
- Define breadth's partner-event distribution, including how roles and multiple
  co-occurrences count, before using it as a covariate.
- Define zero-aware statistical models and probability/logit handling; distinguish
  exposure probability `n50` from total exposure count `N50` at a fixed budget.
- Validate the renderer, rule grammar, image checker and per-rule error rates on
  clean and perturbed renders, then audit generated images with human labels.
- Read and reproduce the closest MOSAIC contrast before making novelty claims.
- Lock pilot choices and exclusions before confirmatory runs. The current
  project has no trained diffusion model that passes a learnability gate.
