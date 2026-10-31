"""Audit a small canonical grammar using exact constraint checks, not sampling.

This module handles the prototype's unary color/shape predicates and strict
left/above relations, with at most three distinct object variables. Feasible
rules come with a witness expressible by the current 3-by-3 scene renderer.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from exposure.render import COLORS, SHAPES
from exposure.rules import Attribute, Predicate, Relation, SceneObject, satisfies


def _order(variables: set[str], edges: list[tuple[str, str]]) -> list[str] | None:
    """Return a deterministic topological order, or None for a strict cycle."""
    remaining = set(variables)
    ordered = []
    while remaining:
        ready = sorted(node for node in remaining if not any(
            target == node and source in remaining for source, target in edges
        ))
        if not ready:
            return None
        node = ready[0]
        ordered.append(node)
        remaining.remove(node)
    return ordered


def assess(predicates: tuple[Predicate, ...]) -> dict:
    """Check exact feasibility and, if feasible, return a three-object witness.

    Proof sketch: conflicting unary assignments and strict order cycles are
    impossible. Otherwise each axis has a topological ordering of at most three
    variables. Placing variables in those orders at grid coordinates 12, 32, 52
    satisfies all relations; distinct x positions also guarantee distinct cells.
    """
    if not predicates:
        raise ValueError("A rule must contain at least one predicate")
    variables: set[str] = set()
    attributes: dict[str, dict[str, str]] = {}
    edges: dict[str, list[tuple[str, str]]] = {"left_of": [], "above": []}
    conflict = False
    for predicate in predicates:
        if isinstance(predicate, Attribute):
            vocabulary = COLORS if predicate.field == "color" else SHAPES
            if predicate.value not in vocabulary:
                raise ValueError(f"Unsupported {predicate.field}: {predicate.value}")
            variables.add(predicate.variable)
            props = attributes.setdefault(predicate.variable, {})
            if predicate.field in props and props[predicate.field] != predicate.value:
                conflict = True
            props[predicate.field] = predicate.value
        elif isinstance(predicate, Relation):
            variables.update((predicate.subject, predicate.object))
            edges[predicate.kind].append((predicate.subject, predicate.object))
        else:
            raise TypeError(f"Unsupported predicate: {predicate!r}")
    if len(variables) > 3:
        return {"feasible": False, "reason": "more_than_three_variables", "witness": None}
    if conflict:
        return {"feasible": False, "reason": "conflicting_attributes", "witness": None}
    x_order = _order(variables, edges["left_of"])
    y_order = _order(variables, edges["above"])
    if x_order is None or y_order is None:
        return {"feasible": False, "reason": "cyclic_relation", "witness": None}
    coordinates = (12, 32, 52)
    scene = []
    for variable in sorted(variables):
        props = attributes.get(variable, {})
        scene.append(SceneObject(props.get("color", "red"), props.get("shape", "circle"),
                                 coordinates[x_order.index(variable)], coordinates[y_order.index(variable)]))
    for i in range(len(variables), 3):
        scene.append(SceneObject("red", "circle", coordinates[i], coordinates[i]))
    if not satisfies(tuple(scene), predicates):
        raise RuntimeError("Internal error: witness does not satisfy rule")
    return {"feasible": True, "reason": None, "witness": [asdict(obj) for obj in scene]}


def canonical_templates(attr_counts=(1, 2, 3), relation_counts=(0, 1, 2)) -> list[dict]:
    """Return representative grammar cells; this is not vocabulary enumeration."""
    if not set(attr_counts) <= {1, 2, 3} or not set(relation_counts) <= {0, 1, 2}:
        raise ValueError("Attribute counts must be 1..3; relation counts must be 0..2")
    attrs = (Attribute("a", "color", "red"), Attribute("b", "shape", "square"),
             Attribute("c", "color", "blue"))
    layouts = {
        0: (("attributes", ()),),
        1: (("single_relation", (Relation("a", "left_of", "b"),)),),
        2: (("chain", (Relation("a", "left_of", "b"), Relation("b", "above", "c"))),
            ("same_pair", (Relation("a", "left_of", "b"), Relation("a", "above", "b")))),
    }
    templates = []
    for n_attrs in sorted(set(attr_counts)):
        for n_rel in sorted(set(relation_counts)):
            for layout, relations in layouts[n_rel]:
                predicates = attrs[:n_attrs] + relations
                encoded = [{"type": "attribute" if isinstance(p, Attribute) else "relation", **asdict(p)}
                           for p in predicates]
                templates.append({"name": f"attr{n_attrs}_rel{n_rel}_{layout}",
                                  "k_attr": n_attrs, "k_rel": n_rel, "layout": layout,
                                  "predicates": encoded, **assess(predicates)})
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a limited canonical scene-rule grammar")
    parser.add_argument("--attrs", nargs="+", type=int, choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--relations", nargs="+", type=int, choices=(0, 1, 2), default=(0, 1, 2))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    templates = canonical_templates(args.attrs, args.relations)
    result = {"scope": "canonical prototype templates, not an exhaustive vocabulary or renderer audit",
              "object_count": 3, "templates": templates}
    content = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
