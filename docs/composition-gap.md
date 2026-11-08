# Composition-gap estimands

`exposure.composition` implements the two composition-gap scales used in the
proposal and guards their boundary behavior. It does not estimate model
performance; it only evaluates a supplied `RSR` against supplied singleton
marginals.

## Definition

For a rule `r` with predicate marginals `c_p`:

```text
Gamma_logit = logit(RSR) - logit(product(c_p))
Gamma_log   = log RSR    - sum(log(c_p))
```

Both are exactly zero when `RSR = product(c_p)` in the interior of the
probability domain. They are different transforms of the same deviation:
`Gamma_logit` is bounded and symmetric, while `Gamma_log` is additive over
predicates. The original review's `logit(product(c_p)) - sum(logit(c_p))` is a
different expression and is not implemented here.

## Boundary behavior

`logit` is undefined at `0` and `1`; `log` is undefined at `0`. The module
returns `None` for an undefined gap and records a `gamma_status`:

| Status | Meaning |
| --- | --- |
| `both_defined` | Both gaps are finite. |
| `log_only` | A marginal is exactly `1`, so `Gamma_logit` is undefined. |
| `undefined` | `RSR` or a marginal is `0`, so neither gap is finite. |

An empirical zero is not proof that a rule is impossible, and an empirical one
is not proof that it is certain. Rule and pair reports from `exposure.mixture`
include `gamma_logit`, `gamma_log`, `gamma_status`, and `marginal_product` for
this reason.

## Runnable commands

```sh
# Evaluate an observed RSR
uv run python -m exposure.composition --rsr 0.42 --marginals 0.8 0.8

# Verify the multiplicative null (both gaps must be zero)
uv run python -m exposure.composition --rsr 0 --marginals 0.8 0.8 --null
```

The `--null` flag ignores `--rsr` and evaluates at
`RSR = product(marginals)`. `tests/test_composition.py` checks the null and the
retracted-expression case.

## Limits

- The product null is a heuristic when predicates share object variables. A
  binding-aware maximum-entropy null is not implemented.
- The module does not propagate measurement uncertainty.
- This is not a Stage 0 gate and says nothing about whether a model learned the
  rule.
