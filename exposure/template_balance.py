"""Describe vocabulary and object-variable coverage across rule-set cells.

Counts are per template, not per predicate occurrence. The audit is descriptive
and does not certify counterbalancing, feasibility, or Stage 0 gates.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from exposure.measure import load_rules, unique_object
from exposure.render import COLORS, SHAPES
from exposure.rules import Attribute, Relation


RELATIONS = ("left_of", "above")
EXPECTED_CELLS = tuple((a, r) for a in (1, 2, 3) for r in (0, 1, 2))


def audit(rules: object) -> dict:
    """Count each vocabulary value at most once per template, per cell."""
    parsed = load_rules(rules)
    grouped: dict[tuple[int, int], dict] = {}
    for predicates in parsed.values():
        attributes = [p for p in predicates if isinstance(p, Attribute)]
        relations = [p for p in predicates if isinstance(p, Relation)]
        cell_key = (len(attributes), len(relations))
        group = grouped.setdefault(cell_key, {"samples": 0, "colors": Counter(),
            "shapes": Counter(), "relations": Counter(), "variables": Counter()})
        group["samples"] += 1
        group["colors"].update({p.value for p in attributes if p.field == "color"})
        group["shapes"].update({p.value for p in attributes if p.field == "shape"})
        group["relations"].update({p.kind for p in relations})
        variables = {p.variable for p in attributes}
        for relation in relations:
            variables.update((relation.subject, relation.object))
        group["variables"][len(variables)] += 1

    cells = []
    for (k_attr, k_rel), group in sorted(grouped.items()):
        samples = group["samples"]
        def coverage(counter, vocabulary):
            return {name: {"count": counter[name], "fraction": counter[name] / samples}
                    for name in vocabulary}
        cells.append({
            "k_attr": k_attr,
            "k_rel": k_rel,
            "template_count": samples,
            "colors": coverage(group["colors"], COLORS),
            "shapes": coverage(group["shapes"], SHAPES),
            "relations": coverage(group["relations"], RELATIONS),
            "object_variable_histogram": {str(n): count for n, count in sorted(group["variables"].items())},
        })
    missing = [{"k_attr": a, "k_rel": r} for a, r in EXPECTED_CELLS if (a, r) not in grouped]
    return {
        "schema_version": 1,
        "scope": "descriptive per-template coverage; not a balance test or feasibility audit",
        "total_templates": len(parsed),
        "cells": cells,
        "missing_expected_cells": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit vocabulary coverage in a named rule set")
    parser.add_argument("--rules", required=True, type=Path, help="JSON mapping of names to predicate lists")
    parser.add_argument("--output", default=Path("outputs/template-balance.json"), type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.rules.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        report = audit(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
