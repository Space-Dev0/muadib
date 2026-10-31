"""Describe local equality directions that preserve singleton scene exposure.

The computation uses the estimated Monte Carlo matrix and does not test
inequalities, nonnegativity, experimental decorrelation, or Stage 0 gates.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from exposure.measure import load_rules, unique_object
from exposure.mixture import summarize
from exposure.rules import Attribute


def _event(predicate) -> tuple[str, ...]:
    if isinstance(predicate, Attribute):
        return ("attribute", predicate.field, predicate.value)
    return ("relation", predicate.kind)


def _event_description(key: tuple[str, ...]) -> dict:
    if key[0] == "attribute":
        return {"type": "attribute", "field": key[1], "value": key[2]}
    return {"type": "relation", "kind": key[1]}


def diagnose(report: dict, weights: dict) -> dict:
    """Compute the nullspace of simplex and unique singleton exposure rows.

    Rank uses max(A.shape) * machine epsilon * the largest singular
    value. Each returned direction is a unit vector in the column order.
    """
    summary = summarize(report, weights)
    rules = load_rules(report["rules"])
    names = list(rules)
    normalized_weights = summary["weights"]
    columns = {column["condition"]: column for column in report["columns"]}
    events: dict[tuple[str, ...], np.ndarray] = {}
    for name, predicates in rules.items():
        for index, predicate in enumerate(predicates):
            key = _event(predicate)
            probabilities = []
            for condition in names:
                cells = columns[condition]["predicate_subsets"][name]
                entry = next(cell for cell in cells if cell["predicate_indices"] == [index])
                probabilities.append(entry["probability"])
            vector = np.asarray(probabilities, dtype=float)
            if key in events and not np.allclose(events[key], vector, rtol=0.0, atol=1e-12):
                raise ValueError(f"Inconsistent singleton exposure vectors for event {key!r}")
            events.setdefault(key, vector)

    ordered_events = sorted(events)
    matrix = np.vstack([np.ones(len(names)), *(events[key] for key in ordered_events)])
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    default_tolerance = max(matrix.shape) * np.finfo(float).eps * float(singular_values[0])
    tolerance = float(default_tolerance)
    rank = int(np.count_nonzero(singular_values > tolerance))
    nullspace = right_vectors[rank:].copy()
    # Fix each basis vector's sign. Degenerate subspaces may still rotate across platforms.
    for vector in nullspace:
        anchor = int(np.argmax(np.abs(vector)))
        if vector[anchor] < 0:
            vector *= -1
    w = np.asarray([normalized_weights[name] for name in names])
    return {
        "schema_version": 1,
        "column_names": names,
        "reference_weights": normalized_weights,
        "minimum_reference_weight": float(np.min(w)),
        "source_seed": report.get("seed"),
        "samples_per_condition": report.get("samples_per_condition"),
        "matrix_shape": list(matrix.shape),
        "simplex_row": [1.0] * len(names),
        "singleton_events": [
            {"event": _event_description(key), "column_probabilities": events[key].tolist(),
             "reference_exposure": float(events[key] @ w)} for key in ordered_events
        ],
        "singular_values": singular_values.tolist(),
        "rank_tolerance": tolerance,
        "rank_tolerance_method": "max(shape)*eps*largest_singular_value",
        "rank": rank,
        "nullity": len(names) - rank,
        "nullspace_basis": nullspace.tolist(),
        "caveats": [
            "This is the nullspace of estimated simplex and singleton equality rows; it is not a decorrelation or Stage 0 test.",
            "Estimated exposures have Monte Carlo uncertainty, which is not propagated.",
            "Directions need not preserve inequality constraints or remain in the simplex, especially at zero-weight boundaries.",
            "Singular-vector bases are not unique and can rotate across numerical platforms in degenerate subspaces.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect singleton-preserving exposure design directions")
    parser.add_argument("--report", type=Path, required=True, help="subset-enabled measurement JSON")
    parser.add_argument("--weights", type=Path, required=True, help="reference condition-weight JSON")
    parser.add_argument("--output", type=Path, default=Path("outputs/design-subspace.json"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        weights = json.loads(args.weights.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = diagnose(report, weights)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
