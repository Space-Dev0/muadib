# Benchmark validity audit

`exposure.benchmark_audit` checks whether a rule set can support a relational
composition contrast. It is adversarial and deterministic unless `--samples` is
given.

```sh
uv run python -m exposure.benchmark_audit \
  --rules outputs/templates-57.json \
  --output outputs/benchmark-audit.json

# Exit with status 2 if the rule set is not confirmatory-ready
uv run python -m exposure.benchmark_audit \
  --rules outputs/templates-57.json \
  --require-confirmatory
```

The rules file is a JSON mapping of names to predicate lists. `--samples`
optionally runs a seeded base-distribution simulation for the saturation check.

## Checks

1. **Relation singleton saturation.** Under existential semantics on the 3×3
   grid with jitter, a single relation is satisfied unless all three objects
   share a column (or row) and all jitters coincide. The analytic exposure is
   `1 - (3/C(9,3))*(1/49) = 0.999271`, above the `0.9` saturation threshold.
2. **Relation-kind/endpoint-binding confound.** Within each `K_rel = 1` cell,
   the audit computes the endpoint-binding class of each relation. A cell is
   flagged when one relation kind is exclusively one class and the other kind
   is not.
3. **Same-object color–shape coverage.** Only color–shape pairs bound to the
   *same* variable can test a conjunction; the audit reports the
   `same_object_pair_count` and the missing pairs.
4. **Held-out template split.** The current rule mapping carries no split
   metadata, so the audit reports `present: false`.

## Current 57-template result

- relation singleton analytic exposure `0.999271` (`saturated: true`);
- `K_attr=2, K_rel=1`: `left_of` is subject-only `3/3`, `above` both-bound
  `3/3`;
- `3/12` same-object color–shape pairs (`red/circle`, `blue/square`,
  `green/triangle`);
- no held-out template split;
- `confirmatory_ready: false`.

## Limits

- The audit detects failures; it does not fix the benchmark.
- It is not a Stage 0 gate, a human audit, or a substitute for generated-image
  validation.
- The saturation result is about the renderer's base distribution, not a
  trained model.
