"""Audit whether a rule set is usable for a relational composition contrast.

The audit is deliberately adversarial.  It checks three properties that the
current 57-template set fails:

* relation singletons are near-tautologies under the renderer's existential
  semantics;
* relation kind is confounded with endpoint binding inside complexity cells;
* color-shape pairs are not fully co-bound to the same object and there is no
  held-out template split.

``confirmatory_ready`` is false until all blocking findings are resolved.  The
module is a diagnostic, not a Stage 0 gate by itself.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from exposure.measure import load_rules, unique_object
from exposure.render import COLORS, SHAPES, _sample_objects
from exposure.rules import Attribute, Relation, satisfies

RELATION_KINDS = ("left_of", "above")
BINDING_CLASSES = ("both", "subject_only", "object_only", "neither")
SATURATION_THRESHOLD = 0.9


def _binding(relation: Relation, bound_variables: set[str]) -> str:
    subject = relation.subject in bound_variables
    target = relation.object in bound_variables
    if subject and target:
        return "both"
    if subject:
        return "subject_only"
    if target:
        return "object_only"
    return "neither"


def _relation_singleton_exposure() -> float:
    """Analytic P(exists a strictly ordered pair) for the 3x3 grid renderer.

    Three distinct grid cells are sampled; failure requires all three in one
    column (3 / C(9,3) = 1/28) and all three x jitters equal (1/49).
    """
    return 1.0 - (3.0 / math.comb(9, 3)) * (1.0 / 49.0)


def _simulate_relation_singletons(samples: int, seed: int) -> dict:
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    counts = Counter()
    import random

    rng = random.Random(seed)
    rules = {kind: (Relation("a", kind, "b"),) for kind in RELATION_KINDS}
    for _ in range(samples):
        scene = _sample_objects(rng)
        for kind, predicates in rules.items():
            counts[kind] += satisfies(scene, predicates)
    return {kind: counts[kind] / samples for kind in RELATION_KINDS}


def audit(rules: object, *, samples: int = 0, seed: int = 0) -> dict:
    """Return a machine-readable benchmark-validity audit."""
    if type(samples) is not int or samples < 0:
        raise ValueError("samples must be a nonnegative integer")
    parsed = load_rules(rules)
    by_cell: dict[tuple[int, int], dict] = {}
    same_object: Counter[tuple[str, str]] = Counter()
    anywhere: Counter[tuple[str, str]] = Counter()
    for name, predicates in parsed.items():
        attributes = [p for p in predicates if isinstance(p, Attribute)]
        relations = [p for p in predicates if isinstance(p, Relation)]
        cell = (len(attributes), len(relations))
        group = by_cell.setdefault(cell, {
            "template_count": 0,
            "by_kind": {kind: Counter() for kind in RELATION_KINDS},
        })
        group["template_count"] += 1
        bound = {p.variable for p in attributes}
        for relation in relations:
            group["by_kind"][relation.kind][_binding(relation, bound)] += 1

        by_variable: dict[str, dict[str, str]] = defaultdict(dict)
        for predicate in attributes:
            by_variable[predicate.variable][predicate.field] = predicate.value
        for values in by_variable.values():
            if "color" in values and "shape" in values:
                same_object[(values["color"], values["shape"])] += 1
        colors = {p.value for p in attributes if p.field == "color"}
        shapes = {p.value for p in attributes if p.field == "shape"}
        for color in colors:
            for shape in shapes:
                anywhere[(color, shape)] += 1

    confounded = []
    for cell, group in sorted(by_cell.items()):
        if cell[1] != 1:
            continue
        for kind in RELATION_KINDS:
            counts = group["by_kind"][kind]
            total = sum(counts.values())
            if total == 0:
                continue
            binding, count = counts.most_common(1)[0]
            if count != total:
                continue
            other = next(item for item in RELATION_KINDS if item != kind)
            other_counts = group["by_kind"][other]
            other_total = sum(other_counts.values())
            share = (other_counts[binding] / other_total) if other_total else 0.0
            if other_total == 0 or share < 1.0:
                confounded.append({
                    "k_attr": cell[0],
                    "k_rel": cell[1],
                    "kind": kind,
                    "binding": binding,
                    "kind_occurrences": total,
                    "other_kind_share": share,
                })

    all_pairs = {(color, shape) for color in COLORS for shape in SHAPES}
    missing_same_object = sorted(all_pairs - set(same_object))
    analytic = _relation_singleton_exposure()
    simulated = _simulate_relation_singletons(samples, seed) if samples > 0 else None
    findings = []
    if analytic > SATURATION_THRESHOLD:
        findings.append({
            "id": "saturated_relation_singletons",
            "severity": "blocking",
            "detail": (f"Analytic single-relation exposure is {analytic:.6f}; a relation "
                       "cannot distinguish relational competence from arbitrary placement."),
        })
    if confounded:
        findings.append({
            "id": "relation_kind_binding_confound",
            "severity": "blocking",
            "detail": (f"{len({(item['k_attr'], item['k_rel']) for item in confounded})} complexity cell(s) assign one relation kind "
                       "exclusively to one endpoint-binding class."),
            "cells": confounded,
        })
    if missing_same_object:
        findings.append({
            "id": "incomplete_same_object_pair_coverage",
            "severity": "blocking",
            "detail": (f"{len(same_object)}/{len(all_pairs)} color-shape pairs are co-bound "
                       "to the same object."),
            "missing": [list(pair) for pair in missing_same_object],
        })
    findings.append({
        "id": "no_held_out_template_split",
        "severity": "blocking",
        "detail": ("The rule set carries no split metadata; the same templates are used "
                   "for training and evaluation."),
    })
    return {
        "schema_version": 1,
        "scope": ("Adversarial benchmark-validity audit; not a Stage 0 gate or a "
                  "substitute for a human or model-output checker."),
        "total_templates": len(parsed),
        "relation_singleton": {
            "analytic_exposure": analytic,
            "simulated_exposure": simulated,
            "threshold": SATURATION_THRESHOLD,
            "saturated": analytic > SATURATION_THRESHOLD,
        },
        "endpoint_binding": {
            "by_cell": {
                f"attr{k_attr}_rel{k_rel}": {
                    "template_count": group["template_count"],
                    "by_kind": {kind: dict(group["by_kind"][kind])
                                for kind in RELATION_KINDS},
                }
                for (k_attr, k_rel), group in sorted(by_cell.items())
            },
            "confounded_cells": confounded,
        },
        "pair_coverage": {
            "same_object_pairs": {f"{color}/{shape}": count
                                  for (color, shape), count in sorted(same_object.items())},
            "anywhere_pair_count": len(anywhere),
            "same_object_pair_count": len(same_object),
            "all_pair_count": len(all_pairs),
            "missing_same_object": [list(pair) for pair in missing_same_object],
        },
        "held_out_split": {
            "present": False,
            "detail": "No split metadata is present in the rule mapping.",
        },
        "blocking_findings": findings,
        "confirmatory_ready": False,
        "note": ("Fix the blocking findings before using this rule set for H1 or H2. "
                 "The audit is deterministic and does not sample unless --samples is given."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a rule set for relational composition validity")
    parser.add_argument("--rules", type=Path, required=True,
                        help="JSON mapping of names to predicate lists")
    parser.add_argument("--samples", type=int, default=0,
                        help="optional base-distribution scenes for a saturation check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-audit.json"))
    parser.add_argument("--require-confirmatory", action="store_true",
                        help="exit with status 2 when the rule set is not confirmatory-ready")
    args = parser.parse_args()
    try:
        rules = json.loads(args.rules.read_text(encoding="utf-8"),
                           object_pairs_hook=unique_object)
        result = audit(rules, samples=args.samples, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")
    if args.require_confirmatory and not result["confirmatory_ready"]:
        parser.error("rule set is not confirmatory-ready: "
                     + ", ".join(item["id"] for item in result["blocking_findings"]))


if __name__ == "__main__":
    main()
