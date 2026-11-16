# Canonical rule grammar audit

Run `uv run python -m exposure.grammar_audit --output outputs/grammar-audit.json`
to inspect a small, deterministic set of prototype rule templates. Optional
`--attrs 1 2 3` and `--relations 0 1 2` restrict the requested complexity cells.

The audit includes one attribute-only and one single-relation template per
attribute count, plus two-relation chain and same-pair layouts. It is a
**representative structural check**, not exhaustive enumeration of colors,
shapes, variable assignments, or the proposal's planned counterbalanced templates.
All cells have at least one feasible template in this restricted set; this does
not prove that every rule in a cell is feasible.

For the prototype's strict `left_of` and `above` relations, feasibility is
decided using conflicting-attribute checks and directed-cycle detection, with
distinct variables limited to three. For an acyclic set, topological orderings
on each axis construct a witness using the renderer's three grid coordinates.
Witnesses include randomized-content-free distractors where needed and are
checked against the rule evaluator. This is an exact test for the currently
supported predicate types and three-object grid, not a probability estimate or
a guarantee that bounded rejection sampling will find a rare rule.

The audit does not test texture, occlusion, a third relation, checker accuracy,
training data balance, generated-image quality, or the proposal's Stage 0
go/no-go criteria. Those remain separate work.
