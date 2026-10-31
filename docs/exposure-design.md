# Exposure Design Feasibility

`exposure.design` checks whether a nonnegative sampling mixture over the
measurement report's conditioning templates can satisfy a set of exposure
bounds. It solves a linear feasibility problem on the **estimated** Monte Carlo
matrix.

> **Warning:** Estimated Monte Carlo feasibility is not population feasibility.
> A feasible solution only satisfies the bounds on the reported probabilities;
> sampling uncertainty is not propagated. An infeasible result only applies to
> the estimated matrix and is not proof that no feasible design exists.

## Runnable commands

Generate a subset-enabled measurement report:

```sh
uv run python -m exposure.measure --rules examples/exposure_rules.json --samples 200 --seed 42 --include-subsets --output outputs/exposure.json
```

Check feasibility with the example constraints:

```sh
uv run python -m exposure.design --report outputs/exposure.json --constraints examples/exposure_constraints.json --output outputs/design.json
```

`--report` and `--constraints` are required. `--output` defaults to
`outputs/design.json`.

## Constraint schema

The constraints file is a JSON object with a nonempty `constraints` list and no
other top-level fields:

```json
{
  "constraints": [
    {"name": "red exposure", "rule": "red", "kind": "predicate", "predicate_indices": [0], "min": 0.45, "max": 1.0},
    {"name": "full rule exposure", "rule": "red_left_of_square", "kind": "rule", "predicate_indices": [], "min": 0.10, "max": 0.9}
  ]
}
```

Each entry has:

- `name`: a unique nonempty constraint name.
- `rule`: a rule name present in the report.
- `kind`: `rule`, `predicate`, or `pair`.
- `predicate_indices`: `[]` for `rule`; one zero-based index for `predicate`;
  two unique ascending valid indices for `pair`.
- `min` and/or `max`: finite numbers in `[0, 1]`, with `min <= max`. At least
  one bound is required.

Entries cannot contain extra fields, and duplicate JSON keys or duplicate
constraint names are rejected. The report must include the original rule
definitions and must have been generated with `--include-subsets`. Malformed
reports, missing subset measurements, and invalid constraints are rejected
before solving.

## Simplex and LP formulation

The report is first validated with `exposure.mixture.summarize` using uniform
weights. That validation checks the schema, subset measurements, and
self-satisfaction cells; the uniform weights are not the design solution.

Let `w_c >= min_weight` be the weight assigned to conditioning template `c`,
where `min_weight` defaults to `0.0`, and let `a_jc` be the measured
probability for constraint row `j` under condition `c`. For a rule constraint,
`a_jc` is the report's full-rule measurement. For a predicate or pair
constraint, it is the corresponding `predicate_subsets` probability.

The solver uses `scipy.optimize.linprog(method="highs")` to solve:

```text
minimize    0^T w
subject to  sum_c w_c = 1
            sum_c a_jc w_c >= min_j   for constraints with a lower bound
            sum_c a_jc w_c <= max_j   for constraints with an upper bound
            w_c >= min_weight
```

All decisions are continuous; there are no integer decisions. The columns of
the matrix correspond to the report's conditioned templates, and the rows
correspond to the requested rule, predicate, or pair exposures.

## Minimum weight floor

`--min-weight` (and the keyword-only `min_weight` argument to `solve`) sets a
minimum weight for every conditioning template. It defaults to `0.0`. The value
must be a finite real number that is nonnegative; booleans and non-finite
values are rejected.

The floor is applied directly as the LP lower bounds `(min_weight, None)` for
all weights. If `min_weight` times the number of conditions exceeds `1`, the
constraint `sum_c w_c = 1` makes the LP infeasible. That case returns
`status: "infeasible"` through the solver; it is not treated as an invalid
report or an invalid constraint specification.

The output includes `min_weight` for both feasible and infeasible statuses.
When a feasible solution falls below the floor, the shortfall is included in
`max_violation`; solutions above the `1e-7` tolerance are treated as numerical
failures. At `min_weight=0.0`, behavior is the same as before the option was
added.

Example with a positive floor:

```sh
uv run python -m exposure.design --report outputs/exposure.json --constraints examples/exposure_constraints.json --min-weight 0.05 --output outputs/design.json
```

## Feasible, infeasible, and solver failure

- **Feasible:** `status` is `"feasible"`. The output includes named `weights`,
  the achieved exposure and slacks for each constraint, and `max_violation`.
- **Infeasible:** `status` is `"infeasible"`, with a `solver_message`. There are
  no fabricated weights or achieved exposures.
- **Solver failure:** any unsuccessful solver status other than infeasibility is
  raised as an error. The CLI reports the error and does not write a design
  output; a numerical failure is never mislabeled as infeasible.

Both feasible and infeasible outputs include the requested `min_weight`.

## Slacks and numerical tolerance

For each constraint, the result reports:

- `exposure`: the achieved weighted probability.
- `lower_slack`: `exposure - min`, or `null` when no lower bound was given.
- `upper_slack`: `max - exposure`, or `null` when no upper bound was given.

Positive slack means the bound is satisfied; negative slack means it is
violated. `max_violation` is the largest of the weight-sum deviation, any
negative weight magnitude, any negative slack, and any shortfall below
`min_weight`. A solution with `max_violation > 1e-7` is treated as a numerical
failure and raises an error rather than being reported as feasible.

The original constraint specification, including the `min` and `max` bounds, is
preserved in the top-level `specification` field. The output also records
`source_seed` (the measurement report's seed), `samples_per_condition`, and the
original `rules`.

## Limits

- The result is feasibility for the estimated matrix, not for the underlying
  population or the true renderer distribution.
- Monte Carlo uncertainty and the measurement Wilson intervals are not
  propagated.
- Fixed marginals must be supplied explicitly, for example with equal `min` and
  `max` bounds; the solver does not choose them automatically.
- The zero objective finds any feasible allocation and can concentrate weight
  on a few conditions. It does not implement the proposal's minimum-KL
  allocation or make causal claims.
- A positive `--min-weight` forces every conditioning template to receive
  weight. It can make the estimated LP infeasible even when a zero-floor
  solution exists; that is not proof of population infeasibility.
- It does not infer infeasibility from the absence of particular realized
  samples. Zero estimated probabilities can make the estimated LP infeasible,
  but that is not proof that the true design is infeasible.
- This is a Stage 0 prototype, not a trained model or the validated exposure
  design.
