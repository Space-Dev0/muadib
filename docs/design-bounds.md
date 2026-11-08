# Exact LP bounds on exposure and lift

`exposure.design_bounds` solves the linear programs behind the singleton-
preserving design problem. `exposure.design_sample` returns a finite sample;
this module distinguishes a narrow sample from a genuinely narrow feasible set.

```sh
uv run python -m exposure.design_bounds \
  --report outputs/exposure-57-pilot.json \
  --weights outputs/uniform-57.json \
  --min-weight 0.0 \
  --output outputs/design-bounds.json
```

The report must include predicate subsets. The module validates the mixture
with `exposure.mixture.summarize`, recomputes the equality nullspace through
`exposure.design_subspace`, and then solves, for each rule:

- minimum and maximum joint exposure;
- the implied log-lift interval (or an explicit boundary status);
- the positive-lift corner and its weights.

All constraints are equalities: simplex plus every unique singleton event fixed
at the reference mixture. `min_weight` is a lower bound on every condition
weight. The reference must already satisfy the floor.

## What the output means

- `lower_exposure` / `upper_exposure` are exact over the estimated matrix.
- `reference_lift` is the lift at the supplied reference weights.
- `lower_lift` / `upper_lift` are `None` with `status` `zero_marginal` or
  `zero_lower_exposure` when a log is undefined.
- `positive_corner` exists only if at least one rule admits strictly positive
  lift; it reports the maximizing weights, the maximum weight, the number of
  weights at the floor, and the support size.
- `summary` reports the number of defined rules, positive-lift rules, min/max
  lift, and median/mean/max lift width.

## Current pilot result

For the 57-rule pilot matrix and uniform reference:

- lift range `[-4.888, +0.337]`;
- median lift width `1.172`;
- `23/57` rules admit positive lift;
- the positive corner places `47/57` weights at the floor.

This is why a narrow sampled range is a sampler/design problem, not proof that
the instrument cannot express positive lift.

## Limits

- Bounds apply to the **estimated** Monte Carlo matrix, not the population.
- Matrix uncertainty is not propagated; these are not confidence bounds.
- The module does not compute a coefficient-targeted (D-optimal or c-optimal)
  design.
- Feasible positive lift at a corner does not mean the effect is practically
  reachable or statistically identified.
