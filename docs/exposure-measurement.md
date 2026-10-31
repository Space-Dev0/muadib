# Exposure Measurement

`exposure.measure` estimates scene-level co-satisfaction for a set of named
rules. It samples accepted scenes for each conditioning rule, then evaluates
every named rule on those scenes. The result describes the prototype renderer,
not training results or a fitted exposure design.

## Runnable CLI example

From the repository root:

```sh
uv run python -m exposure.measure --rules examples/exposure_rules.json --samples 200 --seed 42 --max-attempts 10000 --include-subsets --output outputs/exposure.json
```

This loads the named rule set in `examples/exposure_rules.json`, samples 200
accepted scenes for each condition, and writes the report to
`outputs/exposure.json`.

- `--rules` is required and points to the named JSON rule set.
- `--samples` is the number of accepted scenes per condition; it defaults to
  `1000`.
- `--seed` defaults to `0`.
- `--max-attempts` is the rejection budget per scene; it defaults to `10000`.
- `--include-subsets` also measures singleton and pairwise predicate subsets for
  every rule; it is off by default.
- `--output` defaults to `outputs/exposure.json`.

## Named JSON rule-set format

The rule set is a nonempty JSON object. Each key is a nonempty rule name, and
each value is a nonempty predicate list using the same format as the renderer's
rules:

```json
{
  "red": [
    {"type": "attribute", "variable": "a", "field": "color", "value": "red"}
  ],
  "red_left_of_square": [
    {"type": "attribute", "variable": "a", "field": "color", "value": "red"},
    {"type": "attribute", "variable": "b", "field": "shape", "value": "square"},
    {"type": "relation", "subject": "a", "kind": "left_of", "object": "b"}
  ]
}
```

The key order is preserved. It determines the order of `rule_order` and the
`columns` list, and it affects sampling because the conditions share one random
stream. Duplicate JSON keys are rejected. Predicate order within a rule does
not change satisfaction semantics.

## Report columns and measurements

The report is a matrix. Each column is a conditioning rule, and each measurement
is a measured rule. For a column, `samples` accepted scenes are drawn by
conditioned rejection sampling under that column's rule.

Each column contains:

- `condition`: the name of the rule used to condition sampling for this column.
- `accepted_scenes`: the number of accepted scenes; this is the denominator for
  every measurement in the column.
- `total_attempts`: the sum of rejection attempts used to obtain those accepted
  scenes. It is at least `accepted_scenes`, but it is not a success denominator.
- `measurements`: one entry per named rule with:
  - `successes`: the number of accepted scenes satisfying that measured rule.
  - `probability`: `successes / accepted_scenes`.
  - `wilson_95`: an approximate 95% binomial Wilson interval for that
    probability, clamped to `[0, 1]` and defined at zero and one.

The measurements are **scene satisfaction given the requested condition**, not
model exposure. A condition's own cell is expected to have
`successes == accepted_scenes`, because every accepted scene satisfies the
conditioning rule.

The report also includes `schema_version`, `seed`, `samples_per_condition`,
`include_subsets`, `max_attempts_per_scene`, `interpretation`, `rule_order`,
`columns`, and the original `rules` object. `interpretation` is
`"scene satisfaction given requested condition"`.

## Predicate subsets

With `--include-subsets`, the top-level `include_subsets` flag is `true` and
each column also contains `predicate_subsets`. Each rule name maps to entries
for every singleton and every pair of predicates from that rule. This does not
enumerate triples or other larger subsets.

Each entry has `predicate_indices`, a zero-based list of indices into the
original rule's predicate list. Singles come first in index order, followed by
all pairs. For a three-predicate rule, the order is `[0]`, `[1]`, `[2]`,
`[0, 1]`, `[0, 2]`, `[1, 2]`. Each entry also has `successes`, `probability`,
and `wilson_95`, using the same accepted-scene denominator and interval method
as the full-rule measurements.

Each subset is evaluated separately. Variable identifiers are retained within
a subset, but the variables are existentially bound anew for each subset.
Bindings are not carried over from the full rule or from another subset.
Distinct variables require distinct objects. For example, in a subset
containing both the red attribute for `a` and the relation `a left_of b`, `a`
is the same object across those predicates, `b` must be a different object, and
both bindings are chosen fresh for that subset. A singleton containing only `b`
binds only `b`.

When `--include-subsets` is omitted, `include_subsets` is `false` and no
`predicate_subsets` entries are written.

## Paired measurements and current scope

Within a column, all full-rule measurements and predicate-subset measurements
use the same accepted scenes. They are paired and correlated, not independent
samples. Columns do not share scenes. The report does not yet calculate lift or
other contrasts, does not enforce fixed marginals, and does not validate the
research design.

## Reproducibility and rule ordering

The same seed, sample count, attempt budget, code, dependency versions, and rule
set produce the same report. The rule set's content and key order both matter.
Conditions share a single RNG seeded once and advanced sequentially, so
reordering the rules changes which condition is sampled first and how the random
stream advances for later columns. Reordering can therefore change results, not
just output order.

## Limitations

- **Scene exposure versus condition exposure.** These estimates are synthetic
  scene-satisfaction rates under the renderer's conditioning rules. They are not
  learned model exposures, causal effects, or the proposal's validated exposure
  design. A condition here is a sampling filter, not a model conditioning input.
- **Prototype only.** There is no trained model, image-based checker, or
  linear-programming/exact inference stage. The report describes only the
  prototype renderer's rejection-sampled scenes.
- **Zero is not proof.** Zero successes in a finite sample do not prove that a
  rule is impossible or infeasible. Failure to find a match within the attempt
  budget can mean the rule is rare or infeasible; it does not establish either.
- **No output on failed sampling.** If any condition exhausts its per-scene
  attempt budget, the whole run aborts with an error and writes no new report.
  An existing output file is left untouched. A failed sample is never counted
  as an accepted or negative observation.
- The measurement runs on the CPU only, and pair and marginal exposure semantics
  still need to be locked in the experiment specification before analysis.
