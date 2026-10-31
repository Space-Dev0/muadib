"""Build a deterministic, provisional set of 57 training rule templates.

The ``K_attr = 1`` cells contain seven candidates so the single-attribute
vocabulary covers all four colors and all three shapes.  The ``K_attr = 2`` and
``K_attr = 3`` cells contain six candidates.  This is a hand-designed
instrument-development set, not an exhaustive grammar, a counterbalanced
benchmark, or evidence that the renderer can sample every rule.
"""

import argparse
import json
from collections import Counter
from itertools import permutations
from pathlib import Path

from exposure.grammar_audit import assess
from exposure.measure import load_rules
from exposure.render import COLORS, SHAPES
from exposure.rules import Attribute, Predicate, Relation, SceneObject, satisfies

SCHEMA_VERSION = 1
OBJECT_COUNT = 3
MAX_VARIABLES = 3
ATTRIBUTE_COUNTS = (1, 2, 3)
RELATION_COUNTS = (0, 1, 2)
RELATION_KINDS = ("left_of", "above")
LAYOUT_ORDER = ("attributes", "single_relation", "chain", "same_pair")

# Expected cell sizes and layouts are explicit because K_attr = 1 has one extra
# triangle candidate in every relation cell.  This fixes singleton vocabulary
# coverage; it is not a claim of full counterbalancing.
EXPECTED_CELL_LAYOUT_COUNTS: dict[tuple[int, int], dict[str, int]] = {
    (1, 0): {"attributes": 7},
    (1, 1): {"single_relation": 7},
    (1, 2): {"chain": 4, "same_pair": 3},
    (2, 0): {"attributes": 6},
    (2, 1): {"single_relation": 6},
    (2, 2): {"chain": 3, "same_pair": 3},
    (3, 0): {"attributes": 6},
    (3, 1): {"single_relation": 6},
    (3, 2): {"chain": 3, "same_pair": 3},
}
EXPECTED_CELL_COUNTS = {
    cell: sum(layouts.values()) for cell, layouts in EXPECTED_CELL_LAYOUT_COUNTS.items()
}
EXPECTED_TOTAL = sum(EXPECTED_CELL_COUNTS.values())
EXPECTED_SINGLE_RELATION_SPLITS = {
    1: {"left_of": 3, "above": 4},
    2: {"left_of": 3, "above": 3},
    3: {"left_of": 3, "above": 3},
}

# K_attr = 1 uses seven attribute bundles so singleton rules cover all seven
# distinct values; K_attr = 2 and 3 use six bundles.  Bundles are deliberately
# small and reuse the prototype vocabulary; they are not a complete enumeration.
EXTRA_KATTR1_BUNDLE = (Attribute("a", "shape", "triangle"),)
ATTRIBUTE_BUNDLES: dict[int, tuple[tuple[Attribute, ...], ...]] = {
    1: (
        (Attribute("a", "color", "red"),),
        (Attribute("a", "color", "blue"),),
        (Attribute("a", "color", "green"),),
        (Attribute("a", "color", "yellow"),),
        (Attribute("a", "shape", "circle"),),
        (Attribute("a", "shape", "square"),),
        EXTRA_KATTR1_BUNDLE,
    ),
    2: (
        (Attribute("a", "color", "red"), Attribute("a", "shape", "circle")),
        (Attribute("a", "color", "blue"), Attribute("a", "shape", "square")),
        (Attribute("a", "color", "green"), Attribute("a", "shape", "triangle")),
        (Attribute("a", "color", "yellow"), Attribute("b", "shape", "circle")),
        (Attribute("a", "shape", "square"), Attribute("b", "color", "red")),
        (Attribute("a", "shape", "triangle"), Attribute("b", "shape", "circle")),
    ),
    3: (
        (Attribute("a", "color", "red"), Attribute("a", "shape", "circle"),
         Attribute("b", "color", "blue")),
        (Attribute("a", "color", "blue"), Attribute("a", "shape", "square"),
         Attribute("b", "color", "green")),
        (Attribute("a", "color", "green"), Attribute("a", "shape", "triangle"),
         Attribute("b", "color", "yellow")),
        (Attribute("a", "color", "yellow"), Attribute("b", "shape", "circle"),
         Attribute("c", "color", "red")),
        (Attribute("a", "color", "red"), Attribute("b", "shape", "square"),
         Attribute("c", "color", "blue")),
        (Attribute("a", "color", "blue"), Attribute("b", "shape", "triangle"),
         Attribute("c", "color", "green")),
    ),
}

SINGLE_RELATION_LAYOUTS = (
    ("left_of", (Relation("a", "left_of", "b"),)),
    ("above", (Relation("a", "above", "b"),)),
)

CHAIN_RELATION_LAYOUTS = (
    (Relation("a", "left_of", "b"), Relation("b", "left_of", "c")),
    (Relation("a", "above", "b"), Relation("b", "above", "c")),
    (Relation("a", "left_of", "b"), Relation("b", "above", "c")),
    # Used by the extra K_attr = 1 triangle chain rule.
    (Relation("a", "above", "b"), Relation("b", "left_of", "c")),
)

SAME_PAIR_RELATION_LAYOUTS = (
    (Relation("a", "left_of", "b"), Relation("a", "above", "b")),
    (Relation("a", "left_of", "b"), Relation("b", "above", "a")),
    (Relation("b", "left_of", "a"), Relation("a", "above", "b")),
)

LIMITATIONS = (
    "This is a provisional, hand-designed set, not a counterbalanced or exhaustive benchmark.",
    "The extra K_attr=1 triangle candidates cover all singleton colors and shapes, "
    "but the set does not establish full counterbalancing or gate passage.",
    "Structural feasibility does not establish that rejection sampling can find each rule "
    "within a practical attempt budget; some rules may be too rare.",
    "The set has no held-out split, sampling-support measurement, training result, or Stage 0 pass.",
    "Witnesses are annotation objects returned by grammar_audit.assess, not rendered images.",
    "The vocabulary is limited to three shapes, four colors, left_of, above, and at most three variables.",
)


def _cell_key(k_attr: int, k_rel: int) -> str:
    return f"attr{k_attr}_rel{k_rel}"


def _encode(predicate: Predicate) -> dict:
    if isinstance(predicate, Attribute):
        return {"type": "attribute", "variable": predicate.variable,
                "field": predicate.field, "value": predicate.value}
    return {"type": "relation", "subject": predicate.subject,
            "kind": predicate.kind, "object": predicate.object}


def _variables(predicates: tuple[Predicate, ...]) -> set[str]:
    variables: set[str] = set()
    for predicate in predicates:
        if isinstance(predicate, Attribute):
            variables.add(predicate.variable)
        elif isinstance(predicate, Relation):
            variables.update((predicate.subject, predicate.object))
        else:
            raise TypeError(f"Unsupported predicate: {predicate!r}")
    return variables


def canonical_key(predicates: tuple[Predicate, ...]) -> tuple[tuple[str, ...], ...]:
    """Return an alpha-renaming- and order-invariant semantic key.

    Variables are normalized to ``a``, ``b``, and ``c`` over every bijection.
    Predicates are sorted and duplicate identical predicates are ignored, so
    variable renaming and predicate reordering do not create distinct keys.
    """
    predicates = tuple(predicates)
    if not predicates:
        raise ValueError("A rule must contain at least one predicate")
    variables = _variables(predicates)
    if len(variables) > MAX_VARIABLES:
        raise ValueError("Rules may use at most three distinct variables")
    names = ("a", "b", "c")[:len(variables)]
    ordered_variables = sorted(variables)
    best: tuple[tuple[str, ...], ...] | None = None
    for assignment in permutations(names):
        mapping = dict(zip(ordered_variables, assignment, strict=True))
        atoms = []
        for predicate in predicates:
            if isinstance(predicate, Attribute):
                atoms.append(("attribute", mapping[predicate.variable],
                              predicate.field, predicate.value))
            else:
                atoms.append(("relation", mapping[predicate.subject],
                              predicate.kind, mapping[predicate.object]))
        key = tuple(sorted(set(atoms)))
        if best is None or key < best:
            best = key
    if best is None:
        raise RuntimeError("Internal error: could not canonicalize rule")
    return best


def _relation_variables(relations: list[Relation]) -> set[str]:
    return {variable for relation in relations
            for variable in (relation.subject, relation.object)}


def _is_directed_chain(relations: list[Relation]) -> bool:
    """Return whether two relations form a directed path over three variables."""
    if len(relations) != 2:
        return False
    first, second = relations
    if first.object == second.subject:
        path = (first.subject, first.object, second.object)
    elif second.object == first.subject:
        path = (second.subject, second.object, first.object)
    else:
        return False
    return len(set(path)) == 3


def _layout_matches(layout: str, relations: list[Relation]) -> bool:
    if layout == "attributes":
        return not relations
    if layout == "single_relation":
        return len(relations) == 1
    if layout == "chain":
        return _is_directed_chain(relations)
    if layout == "same_pair":
        return (len(relations) == 2
                and len(_relation_variables(relations)) == 2
                and relations[0] != relations[1])
    return False


def _specifications() -> list[dict]:
    specifications = []
    for k_attr in ATTRIBUTE_COUNTS:
        bundles = ATTRIBUTE_BUNDLES[k_attr]
        for k_rel in RELATION_COUNTS:
            if k_rel == 0:
                for index, bundle in enumerate(bundles, start=1):
                    specifications.append({
                        "name": f"attr{k_attr}_rel0_attributes_{index}",
                        "k_attr": k_attr, "k_rel": k_rel, "layout": "attributes",
                        "predicates": bundle,
                    })
            elif k_rel == 1:
                for index, bundle in enumerate(bundles[:3], start=1):
                    specifications.append({
                        "name": f"attr{k_attr}_rel1_left_of_{index}",
                        "k_attr": k_attr, "k_rel": k_rel, "layout": "single_relation",
                        "predicates": bundle + SINGLE_RELATION_LAYOUTS[0][1],
                    })
                for index, bundle in enumerate(bundles[3:], start=1):
                    specifications.append({
                        "name": f"attr{k_attr}_rel1_above_{index}",
                        "k_attr": k_attr, "k_rel": k_rel, "layout": "single_relation",
                        "predicates": bundle + SINGLE_RELATION_LAYOUTS[1][1],
                    })
            else:
                for index, bundle in enumerate(bundles[:3], start=1):
                    specifications.append({
                        "name": f"attr{k_attr}_rel2_chain_{index}",
                        "k_attr": k_attr, "k_rel": k_rel, "layout": "chain",
                        "predicates": bundle + CHAIN_RELATION_LAYOUTS[index - 1],
                    })
                if k_attr == 1:
                    specifications.append({
                        "name": "attr1_rel2_chain_4",
                        "k_attr": k_attr, "k_rel": k_rel, "layout": "chain",
                        "predicates": EXTRA_KATTR1_BUNDLE + CHAIN_RELATION_LAYOUTS[3],
                    })
                for index, bundle in enumerate(bundles[3:6], start=1):
                    specifications.append({
                        "name": f"attr{k_attr}_rel2_same_pair_{index}",
                        "k_attr": k_attr, "k_rel": k_rel, "layout": "same_pair",
                        "predicates": bundle + SAME_PAIR_RELATION_LAYOUTS[index - 1],
                    })
    return specifications


def _entry(name: str, k_attr: int, k_rel: int, layout: str,
           predicates: tuple[Predicate, ...]) -> dict:
    result = assess(predicates)
    if not result["feasible"]:
        raise ValueError(f"Template {name!r} is not structurally feasible: {result['reason']}")
    witness = [dict(obj) for obj in result["witness"]]
    scene = tuple(SceneObject(**obj) for obj in witness)
    if not satisfies(scene, predicates):
        raise RuntimeError(f"Internal error: witness for {name!r} does not satisfy the rule")
    variables = _variables(predicates)
    kinds = {predicate.kind for predicate in predicates if isinstance(predicate, Relation)}
    colors = {predicate.value for predicate in predicates
              if isinstance(predicate, Attribute) and predicate.field == "color"}
    shapes = {predicate.value for predicate in predicates
              if isinstance(predicate, Attribute) and predicate.field == "shape"}
    return {
        "name": name,
        "k_attr": k_attr,
        "k_rel": k_rel,
        "layout": layout,
        "predicate_count": len(predicates),
        "attribute_count": sum(isinstance(p, Attribute) for p in predicates),
        "relation_count": sum(isinstance(p, Relation) for p in predicates),
        "variable_count": len(variables),
        "relation_kinds": [kind for kind in RELATION_KINDS if kind in kinds],
        "colors": [color for color in COLORS if color in colors],
        "shapes": [shape for shape in SHAPES if shape in shapes],
        "witness": witness,
        "witness_verified": True,
    }


def _manifest(entries: list[dict]) -> dict:
    by_cell = {_cell_key(k_attr, k_rel): 0
               for k_attr in ATTRIBUTE_COUNTS for k_rel in RELATION_COUNTS}
    by_cell_layout = {_cell_key(k_attr, k_rel): {}
                      for k_attr in ATTRIBUTE_COUNTS for k_rel in RELATION_COUNTS}
    by_k_attr: Counter[str] = Counter()
    by_k_rel: Counter[str] = Counter()
    by_layout: Counter[str] = Counter()
    used_colors: set[str] = set()
    used_shapes: set[str] = set()
    used_kinds: set[str] = set()
    used_layouts: set[str] = set()
    used_fields: set[str] = set()
    max_variables = 0

    for entry in entries:
        cell = _cell_key(entry["k_attr"], entry["k_rel"])
        by_cell[cell] += 1
        layout_counts = by_cell_layout[cell]
        layout_counts[entry["layout"]] = layout_counts.get(entry["layout"], 0) + 1
        by_k_attr[str(entry["k_attr"])] += 1
        by_k_rel[str(entry["k_rel"])] += 1
        by_layout[entry["layout"]] += 1
        used_colors.update(entry["colors"])
        used_shapes.update(entry["shapes"])
        used_kinds.update(entry["relation_kinds"])
        used_layouts.add(entry["layout"])
        used_fields.update(
            field for field in ("color", "shape")
            if (field == "color" and entry["colors"])
            or (field == "shape" and entry["shapes"])
        )
        max_variables = max(max_variables, entry["variable_count"])

    ordered_by_layout = {layout: by_layout[layout] for layout in LAYOUT_ORDER
                         if by_layout[layout]}
    ordered_by_cell_layout = {
        cell: {layout: counts[layout] for layout in LAYOUT_ORDER if layout in counts}
        for cell, counts in by_cell_layout.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "provisional hand-designed training rule templates for instrument development",
        "provisional": True,
        "total_templates": len(entries),
        "object_count": OBJECT_COUNT,
        "max_variables": max_variables,
        "colors": list(COLORS),
        "shapes": list(SHAPES),
        "relation_kinds": list(RELATION_KINDS),
        "used": {
            "colors": [color for color in COLORS if color in used_colors],
            "shapes": [shape for shape in SHAPES if shape in used_shapes],
            "relation_kinds": [kind for kind in RELATION_KINDS if kind in used_kinds],
            "layouts": [layout for layout in LAYOUT_ORDER if layout in used_layouts],
            "attribute_fields": [field for field in ("color", "shape") if field in used_fields],
            "max_variables": max_variables,
        },
        "counts": {
            "total": len(entries),
            "by_cell": by_cell,
            "by_k_attr": dict(by_k_attr),
            "by_k_rel": dict(by_k_rel),
            "by_layout": ordered_by_layout,
            "by_cell_layout": ordered_by_cell_layout,
        },
        "witnesses_verified": len(entries),
        "limitations": list(LIMITATIONS),
        "templates": entries,
    }


def validate_templates(rules: object, manifest: object) -> None:
    """Validate parser compatibility, counts, uniqueness, and every witness."""
    if not isinstance(rules, dict) or not rules:
        raise ValueError("Template rules must be a nonempty JSON object")
    parsed = load_rules(rules)
    if len(parsed) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} templates, found {len(parsed)}")
    if not isinstance(manifest, dict):
        raise ValueError("Template manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Template manifest has an unsupported schema version")
    if manifest.get("provisional") is not True:
        raise ValueError("Template manifest must be marked provisional")
    if manifest.get("total_templates") != len(parsed):
        raise ValueError("Manifest total_templates does not match the rule count")
    if manifest.get("object_count") != OBJECT_COUNT:
        raise ValueError(f"Manifest object_count must be {OBJECT_COUNT}")
    if manifest.get("colors") != list(COLORS):
        raise ValueError("Manifest colors do not match the renderer vocabulary")
    if manifest.get("shapes") != list(SHAPES):
        raise ValueError("Manifest shapes do not match the renderer vocabulary")
    if manifest.get("relation_kinds") != list(RELATION_KINDS):
        raise ValueError("Manifest relation kinds do not match the supported relations")

    entries = manifest.get("templates")
    if not isinstance(entries, list) or len(entries) != len(parsed):
        raise ValueError("Manifest templates must match the rule count")
    if [entry.get("name") if isinstance(entry, dict) else None for entry in entries] != list(parsed):
        raise ValueError("Manifest template order must match rule order")

    owners: dict[tuple[tuple[str, ...], ...], str] = {}
    for name, predicates in parsed.items():
        key = canonical_key(predicates)
        if key in owners:
            raise ValueError(f"Duplicate template semantics: {owners[key]!r} and {name!r}")
        owners[key] = name

    by_cell = {_cell_key(k_attr, k_rel): 0
               for k_attr in ATTRIBUTE_COUNTS for k_rel in RELATION_COUNTS}
    by_cell_layout = {_cell_key(k_attr, k_rel): {}
                      for k_attr in ATTRIBUTE_COUNTS for k_rel in RELATION_COUNTS}
    by_cell_kinds = {_cell_key(k_attr, k_rel): Counter()
                     for k_attr in ATTRIBUTE_COUNTS for k_rel in RELATION_COUNTS}
    by_k_attr: Counter[str] = Counter()
    by_k_rel: Counter[str] = Counter()
    by_layout: Counter[str] = Counter()
    used_colors: set[str] = set()
    used_shapes: set[str] = set()
    used_kinds: set[str] = set()
    used_layouts: set[str] = set()
    used_fields: set[str] = set()
    max_variables = 0

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest template must be an object")
        name = entry.get("name")
        if name not in parsed:
            raise ValueError(f"Manifest template {name!r} has no matching rule")
        predicates = parsed[name]
        k_attr, k_rel = entry.get("k_attr"), entry.get("k_rel")
        if k_attr not in ATTRIBUTE_COUNTS or k_rel not in RELATION_COUNTS:
            raise ValueError(f"Template {name!r} has an unsupported cell")
        layout = entry.get("layout")
        if layout not in LAYOUT_ORDER:
            raise ValueError(f"Template {name!r} has an unsupported layout")
        attribute_count = sum(isinstance(p, Attribute) for p in predicates)
        relations = [p for p in predicates if isinstance(p, Relation)]
        relation_count = len(relations)
        if attribute_count != k_attr or relation_count != k_rel:
            raise ValueError(f"Template {name!r} does not match its declared cell")
        if not _layout_matches(layout, relations):
            raise ValueError(f"Template {name!r} has layout {layout!r} "
                             "that does not match its relation graph")
        if entry.get("predicate_count") != len(predicates):
            raise ValueError(f"Template {name!r} has an incorrect predicate_count")
        if entry.get("attribute_count") != attribute_count:
            raise ValueError(f"Template {name!r} has an incorrect attribute_count")
        if entry.get("relation_count") != relation_count:
            raise ValueError(f"Template {name!r} has an incorrect relation_count")
        variables = _variables(predicates)
        if len(variables) > MAX_VARIABLES:
            raise ValueError(f"Template {name!r} uses more than three variables")
        if entry.get("variable_count") != len(variables):
            raise ValueError(f"Template {name!r} has an incorrect variable_count")
        max_variables = max(max_variables, len(variables))

        result = assess(predicates)
        if not result["feasible"]:
            raise ValueError(f"Template {name!r} is not structurally feasible: {result['reason']}")
        witness = entry.get("witness")
        if not isinstance(witness, list) or len(witness) != OBJECT_COUNT:
            raise ValueError(f"Template {name!r} must have a {OBJECT_COUNT}-object witness")
        if witness != result["witness"]:
            raise ValueError(f"Template {name!r} witness does not match the "
                             "deterministic assess witness")
        try:
            scene = tuple(SceneObject(**obj) for obj in witness)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Template {name!r} has a malformed witness") from exc
        if not satisfies(scene, predicates):
            raise ValueError(f"Template {name!r} witness does not satisfy the rule")
        if entry.get("witness_verified") is not True:
            raise ValueError(f"Template {name!r} witness is not marked verified")

        kinds = {p.kind for p in predicates if isinstance(p, Relation)}
        relation_kinds = [kind for kind in RELATION_KINDS if kind in kinds]
        colors = [color for color in COLORS
                  if color in {p.value for p in predicates
                               if isinstance(p, Attribute) and p.field == "color"}]
        shapes = [shape for shape in SHAPES
                  if shape in {p.value for p in predicates
                               if isinstance(p, Attribute) and p.field == "shape"}]
        if entry.get("relation_kinds") != relation_kinds:
            raise ValueError(f"Template {name!r} has incorrect relation_kinds")
        if entry.get("colors") != colors or entry.get("shapes") != shapes:
            raise ValueError(f"Template {name!r} has incorrect color or shape metadata")

        cell = _cell_key(k_attr, k_rel)
        by_cell[cell] += 1
        layout_counts = by_cell_layout[cell]
        layout_counts[layout] = layout_counts.get(layout, 0) + 1
        for kind in relation_kinds:
            by_cell_kinds[cell][kind] += 1
        by_k_attr[str(k_attr)] += 1
        by_k_rel[str(k_rel)] += 1
        by_layout[layout] += 1
        used_colors.update(colors)
        used_shapes.update(shapes)
        used_kinds.update(relation_kinds)
        used_layouts.add(layout)
        used_fields.update(
            field for field in ("color", "shape")
            if (field == "color" and colors) or (field == "shape" and shapes)
        )

    for (k_attr, k_rel), expected_count in EXPECTED_CELL_COUNTS.items():
        cell = _cell_key(k_attr, k_rel)
        if by_cell[cell] != expected_count:
            raise ValueError(f"Cell {cell!r} must contain {expected_count} templates")
    for (k_attr, k_rel), expected_layouts in EXPECTED_CELL_LAYOUT_COUNTS.items():
        cell = _cell_key(k_attr, k_rel)
        if by_cell_layout[cell] != expected_layouts:
            raise ValueError(f"Cell {cell!r} must have layout counts {expected_layouts!r}")
    for k_attr, expected_kinds in EXPECTED_SINGLE_RELATION_SPLITS.items():
        cell = _cell_key(k_attr, 1)
        actual_kinds = {kind: by_cell_kinds[cell].get(kind, 0)
                        for kind in ("left_of", "above")}
        if actual_kinds != expected_kinds:
            raise ValueError(f"Cell {cell!r} must have relation-kind counts {expected_kinds!r}")

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Manifest must contain counts")
    if counts.get("total") != len(parsed):
        raise ValueError("Manifest total count is incorrect")
    if counts.get("by_cell") != by_cell:
        raise ValueError("Manifest per-cell counts are incorrect")
    if counts.get("by_k_attr") != dict(by_k_attr):
        raise ValueError("Manifest per-K_attr counts are incorrect")
    if counts.get("by_k_rel") != dict(by_k_rel):
        raise ValueError("Manifest per-K_rel counts are incorrect")
    expected_by_layout = {layout: by_layout[layout] for layout in LAYOUT_ORDER
                          if by_layout[layout]}
    if counts.get("by_layout") != expected_by_layout:
        raise ValueError("Manifest per-layout counts are incorrect")
    expected_by_cell_layout = {
        cell: {layout: values[layout] for layout in LAYOUT_ORDER if layout in values}
        for cell, values in by_cell_layout.items()
    }
    if counts.get("by_cell_layout") != expected_by_cell_layout:
        raise ValueError("Manifest per-cell layout counts are incorrect")
    if manifest.get("max_variables") != max_variables:
        raise ValueError("Manifest max_variables is incorrect")
    if manifest.get("witnesses_verified") != len(parsed):
        raise ValueError("Manifest witnesses_verified is incorrect")
    expected_used = {
        "colors": [color for color in COLORS if color in used_colors],
        "shapes": [shape for shape in SHAPES if shape in used_shapes],
        "relation_kinds": [kind for kind in RELATION_KINDS if kind in used_kinds],
        "layouts": [layout for layout in LAYOUT_ORDER if layout in used_layouts],
        "attribute_fields": [field for field in ("color", "shape") if field in used_fields],
        "max_variables": max_variables,
    }
    if manifest.get("used") != expected_used:
        raise ValueError("Manifest used-vocabulary summary is incorrect")


def build_templates() -> tuple[dict[str, list[dict]], dict]:
    """Return validated rules and their audit manifest."""
    rules: dict[str, list[dict]] = {}
    entries = []
    for specification in _specifications():
        predicates = tuple(specification["predicates"])
        rules[specification["name"]] = [_encode(p) for p in predicates]
        entries.append(_entry(specification["name"], specification["k_attr"],
                              specification["k_rel"], specification["layout"], predicates))
    manifest = _manifest(entries)
    validate_templates(rules, manifest)
    return rules, manifest


def _resolve_output_paths(parser: argparse.ArgumentParser, rules_output: Path,
                          manifest_output: Path) -> tuple[Path, Path]:
    """Resolve both output paths and reject aliases before any write."""
    rules_path = rules_output.expanduser().resolve()
    manifest_path = manifest_output.expanduser().resolve()
    same = rules_path == manifest_path
    if not same and rules_path.exists() and manifest_path.exists():
        try:
            same = rules_path.samefile(manifest_path)
        except OSError:
            same = False
    if same:
        parser.error("--rules-output and --manifest-output must resolve to different files")
    return rules_path, manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the provisional 57-template training rule set")
    parser.add_argument("--rules-output", type=Path, default=Path("outputs/templates.json"),
                        help="path for the JSON mapping of names to predicate lists")
    parser.add_argument("--manifest-output", type=Path,
                        default=Path("outputs/templates_manifest.json"),
                        help="path for the audit manifest")
    args = parser.parse_args(argv)
    rules_path, manifest_path = _resolve_output_paths(
        parser, args.rules_output, args.manifest_output)

    rules, manifest = build_templates()
    rules_content = json.dumps(rules, indent=2) + "\n"
    manifest_content = json.dumps(manifest, indent=2) + "\n"
    # Validate the serialized forms before creating any output file.
    validate_templates(json.loads(rules_content), json.loads(manifest_content))

    rules_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text(rules_content, encoding="utf-8")
    manifest_path.write_text(manifest_content, encoding="utf-8")
    print(f"Wrote {rules_path} and {manifest_path}")


if __name__ == "__main__":
    main()
