# Rendering and Rules

The renderer creates annotated synthetic scenes for instrument development. It
can sample unconditionally or condition on a rule. This is a prototype, not the
validated exposure design, and it runs on the CPU only.

## Commands

Render without a rule:

```sh
uv run python -m exposure.render --seed 42 --output outputs/example
```

Render with a rule:

```sh
uv run python -m exposure.render --seed 42 --rule examples/red_left_of_square.json --output outputs/conditioned
```

`--rule` is optional. `--seed` defaults to `0`, `--output` defaults to
`outputs/scene`, and `--max-attempts` defaults to `10000`. The output prefix
produces both a `.png` image and a `.json` annotation file.

## Rule files

A rule file is a nonempty JSON list of predicates. There are two predicate
types.

An attribute predicate has exactly these fields:

```json
{"type": "attribute", "variable": "a", "field": "color", "value": "red"}
```

- `field` is `color` or `shape`.
- Supported colors are `red`, `blue`, `green`, and `yellow`.
- Supported shapes are `circle`, `square`, and `triangle`.

A relation predicate has exactly these fields:

```json
{"type": "relation", "subject": "a", "kind": "left_of", "object": "b"}
```

- `kind` is `left_of` or `above`.
- `left_of` and `above` compare object centers.
- `subject` and `object` must be different variables.

The example rule requests a red object to the left of a square:

```json
[
  {"type": "attribute", "variable": "a", "field": "color", "value": "red"},
  {"type": "attribute", "variable": "b", "field": "shape", "value": "square"},
  {"type": "relation", "subject": "a", "kind": "left_of", "object": "b"}
]
```

## Predicate semantics

Variables are distinct and existentially bound. A scene satisfies a rule when
some assignment of distinct objects to the rule's variables satisfies every
predicate. A shared variable name keeps the same binding across predicates, and
two different variables cannot refer to the same object.

Coordinates follow the image convention: `x` increases rightward and `y`
increases downward. `left_of` requires the subject's center to have a smaller
`x`, and `above` requires it to have a smaller `y`. Ties do not satisfy either
relation.

Generated object positions are bounding-box centers. The renderer draws each
shape around its center, and the annotation's `x` and `y` are that center.
Evaluating a subset of a rule independently existentially quantifies only the
variables present in that subset. Pair and marginal exposure semantics still
need to be locked in the experiment specification before analysis.

## Sampling

The renderer samples three objects from a grid-based distribution: it chooses
three distinct cells from a 3x3 grid, jitters each position slightly, and draws
a random supported color and shape. Shapes are separated and drawn with
antialiased edges, then the scene is recorded as a 64x64 RGB image.

Conditioned rendering uses **bounded rejection sampling**. It draws scenes from
the same three-object distribution until the annotations satisfy the rule.
`--max-attempts` sets the budget and defaults to `10000`. Failure within the
budget can mean the rule is rare or infeasible; it does not prove
infeasibility. `--max-attempts` must be positive, and a rule must contain at
least one predicate.

## Output JSON

Every run writes `seed`, `image_size`, and `objects`. Each object records
`color`, `shape`, `x`, and `y`.

When a rule is supplied, the JSON also includes the requested `rule`, the number
of `attempts` used, and `max_attempts`. An unconditional render omits those
three fields.

Seeded output is reproducible when the code and dependency versions match.
The prototype does not yet build masks or implement the validated exposure
design.
