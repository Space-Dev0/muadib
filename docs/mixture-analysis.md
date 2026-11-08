# Mixture Analysis

`exposure.mixture` combines a subset-enabled measurement report under a
specified sampling mixture. It computes weighted scene-satisfaction estimates
and log lifts from the prototype renderer's measurements. These are descriptive
Monte Carlo plug-in estimates, not results from a trained model or a validated
exposure design.

## Runnable commands

First create a measurement report with full-rule and subset measurements:

```sh
uv run python -m exposure.measure --rules examples/exposure_rules.json --samples 200 --seed 42 --include-subsets --output outputs/exposure.json
```

Then combine that report using the uniform weights:

```sh
uv run python -m exposure.mixture --report outputs/exposure.json --weights examples/uniform_weights.json --output outputs/mixture.json
```

The first command samples 200 accepted scenes per condition. The second writes
the weighted analysis to `outputs/mixture.json`. `--report` and `--weights` are
required; `--output` defaults to `outputs/mixture.json`.

The measurement report must be schema version 1 and must have been generated
with `--include-subsets`. The mixture command rejects reports that lack the
required subset measurements.

## Weight file

The weights file is a JSON object mapping every condition name to a number:

```json
{
  "red": 0.25,
  "square": 0.25,
  "red_left_of_square": 0.25,
  "blue_above_circle": 0.25
}
```

Weights must contain exactly one entry for each condition, with no missing or
extra names. Every weight must be a finite nonnegative number, and the weights
must sum to 1 (within a small numerical tolerance). Duplicate JSON keys are
rejected. The output records the normalized weights in the report's rule order.

## Weighted exposure and log lift

For a condition `c` with weight `w_c` and a measured rule `S` with probability
`p_c(S)`, the weighted exposure is:

```text
n(S) = sum_c w_c * p_c(S)
```

This is reported as `rule_exposure` for each rule.

For each predicate `i` in a rule, the weighted singleton probability is:

```text
n_i = sum_c w_c * p_c({i})
```

These values are reported in `predicate_marginals`.

For each predicate pair `(i, j)`, the weighted pair probability is:

```text
n_ij = sum_c w_c * p_c({i, j})
```

These are reported in `pairs` as `exposure`, with their zero-based
`predicate_indices`.

The per-rule log lift is the natural logarithm of the joint weighted exposure
divided by the product of its weighted predicate marginals:

```text
log_lift(S) = log(n(S) / product_i n_i)
```

The pair log lift is:

```text
log_lift(i, j) = log(n_ij / (n_i * n_j))
```

A log lift is `null` when the joint or any required marginal is zero. The
`lift_status` field distinguishes `zero_joint` (the empirical joint is zero
while the marginals are positive), `zero_marginal`, and `defined`. With a zero
joint and positive marginals the empirical log lift is negative infinity;
`null` keeps the JSON finite. An empirical zero is not proof that the rule or
pair is impossible.

Each rule and pair also reports two composition gaps derived from the same
marginals:

```text
gamma_logit = logit(n(S)) - logit(product_i n_i)
gamma_log   = log(n(S)) - sum_i log(n_i)
```

Both are zero under `n(S) = product_i n_i`. These are **exposure** diagnostics,
not generated-image rule-satisfaction rates or the proposal's outcome gap.
`gamma_logit` is `null` when the joint or product is `0` or `1`;
`gamma_log` is `null` when the joint or any marginal is `0`. The
`gamma_status` field records `both_defined`, `log_only`, or `undefined`.
`marginal_product` is reported alongside them. See
[composition gap](composition-gap.md).

## Subset semantics

The mixture step consumes the `predicate_subsets` produced by
`exposure.measure`; it does not re-bind or reinterpret them. Within each subset,
variable identifiers are retained across that subset's predicates, but the
variables are existentially bound anew for each subset. Bindings are not carried
over from the full rule or from another subset, and distinct variables require
distinct objects. A singleton is therefore evaluated from scratch, not as a
projection of the full-rule assignment.

In particular, both singleton events can hold because of the same object even
when a pair with distinct variables fails. The pair exposure is evaluated with
shared bindings and distinctness within the pair; it need not equal the
probability that the separately evaluated singleton events both hold. The
product of the singleton probabilities is therefore a heuristic reference.
Mixing conditioned distributions also changes these statistics, by design.

## Interpretation and limitations

- **No uncertainty propagation.** The mixture output has no confidence
  intervals. It uses the measurement probabilities and does not propagate the
  measurement's Wilson intervals.
- **Pointwise, not simultaneous.** Wilson intervals in the measurement report
  are pointwise intervals for individual cells. They are not simultaneous
  intervals for the whole matrix, and they should not be combined as though the
  cells were independent.
- **No fixed marginals.** The analysis reweights conditions but does not enforce
  fixed predicate marginals or construct a controlled exposure design.
- **Descriptive, not causal.** The estimates describe the prototype renderer's
  rejection-sampled scenes. They are not proof of causality and do not support
  claims about a trained model or a validated design.
- **Known self-satisfaction cells.** A condition's own cell is known by
  construction to be exactly `1`, i.e. the interval `[1, 1]`, not a Wilson
  interval. The measurement report emits `[1, 1]` in its `wilson_95` field
  for that cell and for its subsets; these are not independent evidence about
  the known events.
- **Order and reproducibility.** Measurement sampling uses a single RNG in rule
  order, so reordering rules changes the sampled scenes and results. Seeded
  images may also differ across architectures or library/code versions.
  Reproduce results with the same rule order, code, dependency versions, and
  platform.
