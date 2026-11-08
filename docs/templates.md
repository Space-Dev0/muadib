# Provisional training rule templates

`exposure.templates` builds a deterministic, explicitly provisional set of 57
named training rule templates. The proposal asks for about 54 rather than an
exact count. The set has seven candidates in each `K_attr = 1` cell and six in
each `K_attr = 2` or `K_attr = 3` cell, covering `K_rel = 0, 1, 2`.

The extra `K_attr = 1` candidate in each relation cell has a single `triangle`
attribute predicate, plus the cell's relation predicates where applicable.
Without it, the six singleton templates in a cell could not cover all seven
distinct unary values (four colors plus three shapes), so `triangle` was missing
from every `K_attr = 1` cell. This fixes that vocabulary gap. It does not
establish full counterbalancing, sampling support, or gate passage.

This is an instrument-development set. It is not an exhaustive grammar, a
counterbalanced benchmark, a held-out split, or evidence that the renderer can
sample every rule within a practical rejection budget.

## Generate the files

```sh
uv run python -m exposure.templates \
  --rules-output outputs/templates.json \
  --manifest-output outputs/templates_manifest.json
```

`--rules-output` defaults to `outputs/templates.json` and
`--manifest-output` defaults to `outputs/templates_manifest.json`. Both paths
are expanded and resolved before any write. If they resolve to the same file,
including aliases such as `.`/`..` components, redundant separators, symlinks,
or existing hard links, the command exits with an error and writes nothing.
Both files are written only after the in-memory templates and their serialized
JSON forms have passed validation.

The rules file is a JSON object mapping each template name to a predicate list
in the format accepted by `exposure.measure` and `exposure.render.parse_rule`:

```json
{
  "attr2_rel2_chain_1": [
    {"type": "attribute", "variable": "a", "field": "color", "value": "red"},
    {"type": "attribute", "variable": "a", "field": "shape", "value": "circle"},
    {"type": "relation", "subject": "a", "kind": "left_of", "object": "b"},
    {"type": "relation", "subject": "b", "kind": "left_of", "object": "c"}
  ]
}
```

The manifest is separate from the rules and records the audit information:
per-cell counts, vocabulary, layouts, object count, witnesses, and limitations.

## Cell and layout construction

`K_attr = 1` cells contain seven candidates; `K_attr = 2` and `K_attr = 3`
cells contain six:

- `K_rel = 0`: attribute-only bundles. `K_attr = 1` has seven bundles, named
  `attr1_rel0_attributes_{1..7}`, including `triangle` as the seventh. The
  other `K_attr` cells have six bundles named
  `attr{K_attr}_rel0_attributes_{1..6}`.
- `K_rel = 1`: rules using one relation, all with layout `single_relation`.
  `K_attr = 1` has three `left_of` rules and four `above` rules, so its split is
  3/4; the extra rule is `attr1_rel1_above_4` (triangle). The `K_attr = 2` and
  `K_attr = 3` cells each have three `left_of` and three `above` rules. Names
  are `attr{K_attr}_rel1_left_of_{1..3}` and
  `attr{K_attr}_rel1_above_{1..4}` for `K_attr = 1`, or
  `attr{K_attr}_rel1_above_{1..3}` otherwise.
- `K_rel = 2`: chain rules using two relations over three variables
  (`a left_of b, b left_of c`; `a above b, b above c`;
  `a left_of b, b above c`; or for the extra triangle candidate
  `a above b, b left_of c`) and same-pair rules using two relations over one
  pair (`a left_of b, a above b`; `a left_of b, b above a`; or
  `b left_of a, a above b`). `K_attr = 1` has a 4/3 chain/same-pair split, so
  its extra rule is `attr1_rel2_chain_4` (triangle). The `K_attr = 2` and
  `K_attr = 3` cells each have a 3/3 split. Names are
  `attr{K_attr}_rel2_chain_{1..4}` for `K_attr = 1` or `{1..3}` otherwise,
  and `attr{K_attr}_rel2_same_pair_{1..3}`.

The attribute bundles use the renderer's existing four colors and three shapes.
Every `K_attr = 1` cell contains all four colors and all three shapes across its
seven templates. Relations use only `left_of` and `above`, and every rule uses
at most three distinct variables.

The per-cell sizes are therefore 7/7/7 for `K_attr = 1` and 6/6/6 for
`K_attr = 2, 3`, for 57 total. The manifest records these counts explicitly.

Candidate order is fixed: ascending `K_attr`, then ascending `K_rel`, then the
layout and candidate index shown above. JSON object order therefore reproduces
the same output on every run.

## Validation

Before returning or writing anything, the builder checks:

- exactly 57 templates: seven in each `K_attr = 1` cell and six in each
  `K_attr = 2` or `K_attr = 3` cell;
- the expected `K_rel = 2` layout splits: 4 chain/3 same-pair for
  `K_attr = 1`, and 3/3 for `K_attr = 2` and `K_attr = 3`;
- the expected `K_rel = 1` relation-kind splits: 3 `left_of`/4 `above` for
  `K_attr = 1`, and 3/3 for `K_attr = 2` and `K_attr = 3`;
- parser compatibility through `exposure.measure.load_rules`;
- no semantic duplicates after normalizing variable renaming and predicate
  order, including duplicate identical predicates;
- structural feasibility through `exposure.grammar_audit.assess`;
- the declared layout matches the actual relation graph: `attributes` has no
  relations, `single_relation` has exactly one, `chain` has two directed
  relations forming a path over exactly three variables with the shared middle
  endpoint, and `same_pair` has two distinct relations over the same two
  variables. Swapping layout labels within a cell therefore fails even when
  aggregate counts are unchanged;
- a three-object witness for every template, checked by
  `exposure.rules.satisfies` and required to equal the deterministic witness
  returned by `exposure.grammar_audit.assess(predicates)`. This rejects
  overlapping, out-of-bounds, or otherwise noncanonical witnesses even when
  they happen to satisfy the rule;
- manifest counts, vocabulary, layouts, and witness totals match the rules.

`canonical_key` renames variables to `a`, `b`, and `c` over every bijection,
sorts predicates, and removes repeated identical predicates. It detects
duplicate atoms under variable renaming, predicate reordering, and idempotence
of repeated identical predicates. It does **not** detect arbitrary logical
equivalence, implication, or redundancy between different predicates; it is not
a theorem prover.

## Benchmark validity

Structural validation is necessary but not sufficient. The loader now also
rejects duplicate identical predicates, contradictory attribute values on the
same variable, and cyclic strict relations through
`exposure.rules.validate_rule` (applied by `render.parse_rule`).

`exposure.benchmark_audit` checks three properties that this provisional set
fails:

- relation singleton exposure is `0.999271`, so a relation cannot separate
  relational competence from arbitrary placement;
- relation kind is confounded with endpoint binding: in `K_attr=2, K_rel=1`,
  `left_of` is subject-only `3/3` and `above` both-bound `3/3`;
- only `3/12` color–shape pairs are co-bound to the same object, and there is no
  held-out template split.

The audit sets `confirmatory_ready: false` and is not a Stage 0 gate by itself.
See [benchmark audit](benchmark-audit.md).

## Manifest fields

The manifest includes:

- `schema_version`, `scope`, `provisional`, and `limitations`;
- `total_templates` (57), `object_count`, and `max_variables`;
- `colors`, `shapes`, and `relation_kinds`, which are the supported vocabulary;
- `used`, which reports the colors, shapes, relation kinds, layouts, attribute
  fields, and maximum variable count that actually occur;
- `counts.by_cell`, `counts.by_k_attr`, `counts.by_k_rel`,
  `counts.by_layout`, and `counts.by_cell_layout`;
- `witnesses_verified`;
- `templates`, with per-template cell, layout, predicate counts, variable count,
  used colors/shapes/relations, and the verified witness.

## Use with the measurement prototype

The rules file can be passed to the existing measurement command:

```sh
uv run python -m exposure.measure \
  --rules outputs/templates.json \
  --samples 200 --seed 42 --include-subsets \
  --output outputs/templates-exposure.json
```

A successful measurement run is still only a Monte Carlo estimate for this
prototype renderer. It is not sampling-support evidence for the population.

## Limitations

- The set is hand-designed and provisional. It is not exhaustive and does not
  provide full counterbalancing of colors, shapes, variables, or layouts.
- Relation kind is confounded with endpoint binding within complexity cells, and
  only `3/12` color–shape pairs are co-bound to the same object; see
  [benchmark audit](benchmark-audit.md).
- Adding the triangle singleton candidates fixes missing vocabulary coverage in
  the `K_attr = 1` cells. It does not establish full counterbalancing, balanced
  frequencies, sampling support, or a Stage 0 gate pass.
- Structural feasibility only means a witness annotation exists. Some rules may
  be too rare for the current bounded rejection sampler.
- The set does not include held-out templates, training results, an independent
  image-checker audit, or a Stage 0 pass.
- Witnesses are annotation objects from `grammar_audit.assess`, not rendered
  images. They do not establish detector performance or human interpretability.
- The grammar is limited to three shapes, four colors, `left_of`, `above`, and
  at most three variables.
