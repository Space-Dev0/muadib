# Singleton-preserving design subspace

Given a subset-enabled measurement report and a condition-weight JSON file:

```sh
uv run python -m exposure.design_subspace --report outputs/exposure-subsets.json --weights examples/uniform_weights.json --output outputs/design-subspace.json
```

The diagnostic validates both inputs with `exposure.mixture.summarize`. It
groups singleton scene events by attribute `(field, value)` or relation kind,
regardless of variable name: a one-predicate existential event has the same
meaning under variable renaming. Repeated instances must have identical
estimated probabilities across conditions (within `1e-12`), or the run fails.

The matrix has one column per condition, an all-ones simplex row, and one row
per unique singleton event. The output includes reference weights and singleton
exposures, column probabilities, singular values, rank, nullity, tolerance,
minimum reference weight, and a nullspace basis. Each basis row is a unit
direction that preserves the estimated simplex and singleton equalities. Rank
uses the absolute threshold `max(matrix.shape) * machine_epsilon * largest`
`singular_value`. Values at or below the threshold are treated as zero.

These directions describe an **affine equality subspace** only. They may leave
the nonnegative simplex, particularly at zero-weight boundaries, or violate
other constraints. They do not establish lift decorrelation, VIF criteria, or a
Stage 0 pass. Matrix sampling uncertainty is not propagated. SVD bases for
degenerate subspaces are not unique and can rotate across numerical platforms;
the subspace is the meaningful object, not a particular basis vector.
