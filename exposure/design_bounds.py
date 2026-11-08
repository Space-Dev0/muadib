"""Exact LP bounds on rule exposure and composition lift.

The hit-and-run sampler in :mod:`exposure.design_sample` returns a finite
sample.  This module solves the underlying linear programs instead, so a narrow
sample can be distinguished from a genuinely narrow feasible set.

The constraints are the simplex and the estimated singleton-exposure rows at a
reference mixture.  For each rule the module reports the minimum and maximum
achievable joint exposure, the implied log-lift interval, and the weights that
attain the largest positive lift.  All bounds apply to the estimated Monte
Carlo matrix; they are not population bounds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from exposure.design_subspace import _event, diagnose
from exposure.measure import load_rules, unique_object
from exposure.mixture import summarize


def _solver_row(report: dict, rule: str, names: list[str]) -> np.ndarray:
    columns = {column["condition"]: column for column in report["columns"]}
    return np.asarray([columns[name]["measurements"][rule]["probability"]
                       for name in names], dtype=float)


def _marginal_product(rule_predicates, events: dict[tuple[str, ...], np.ndarray],
                      reference: np.ndarray) -> tuple[float, list[float]]:
    marginals = []
    for predicate in rule_predicates:
        key = _event(predicate)
        if key not in events:
            raise ValueError(f"Missing singleton event {key!r}")
        marginals.append(float(events[key] @ reference))
    if any(value <= 0.0 for value in marginals):
        return 0.0, marginals
    return math.prod(marginals), marginals


def _lift_bounds(lower: float, upper: float, product: float,
                 marginals: list[float]) -> tuple[float | None, float | None, str]:
    """Return lift bounds, or None at an empirical boundary."""
    if product <= 0.0 or any(value <= 0.0 for value in marginals):
        return None, None, "zero_marginal"
    if lower <= 0.0:
        base = math.fsum(math.log(value) for value in marginals)
        return None, math.log(upper) - base if upper > 0 else None, "zero_lower_exposure"
    base = math.fsum(math.log(value) for value in marginals)
    return math.log(lower) - base, math.log(upper) - base, "defined"


def feasible_lift_bounds(report: dict, weights: dict, *,
                         min_weight: float = 0.0) -> dict:
    """Solve exact exposure and lift bounds over the singleton-preserving set.

    ``weights`` is the reference mixture.  The equality constraints fix every
    unique singleton event at its reference exposure, exactly as
    :mod:`exposure.design_subspace` and :mod:`exposure.design_sample` do.
    """
    if (isinstance(min_weight, bool) or not isinstance(min_weight, (int, float))
            or not math.isfinite(min_weight) or min_weight < 0):
        raise ValueError("min_weight must be a finite nonnegative number")
    summarize(report, weights)
    subspace = diagnose(report, weights)
    names = list(subspace["column_names"])
    rules = load_rules(report["rules"])
    if list(rules) != names:
        raise ValueError("Rule order changed between measurement and subspace")
    reference = np.asarray([subspace["reference_weights"][name] for name in names],
                           dtype=float)
    if np.any(reference < min_weight - 1e-12):
        raise ValueError("Reference weights must satisfy the minimum-weight floor")

    event_vectors: dict[tuple[str, ...], np.ndarray] = {}
    event_rows = []
    for item in subspace["singleton_events"]:
        description = item["event"]
        if description["type"] == "attribute":
            key = ("attribute", description["field"], description["value"])
        else:
            key = ("relation", description["kind"])
        vector = np.asarray(item["column_probabilities"], dtype=float)
        event_vectors[key] = vector
        event_rows.append(vector)
    matrix = np.vstack([np.asarray(subspace["simplex_row"], dtype=float), *event_rows])
    target = matrix @ reference
    bounds = [(min_weight, None)] * len(names)

    rows = []
    for rule, predicates in rules.items():
        row = _solver_row(report, rule, names)
        lower_result = linprog(row, A_eq=matrix, b_eq=target, bounds=bounds,
                               method="highs")
        upper_result = linprog(-row, A_eq=matrix, b_eq=target, bounds=bounds,
                               method="highs")
        if lower_result.status == 2 or upper_result.status == 2:
            rows.append({"rule": rule, "status": "infeasible"})
            continue
        if not lower_result.success or not upper_result.success:
            message = lower_result.message if not lower_result.success else upper_result.message
            raise RuntimeError(f"Linear solver failed for {rule!r}: {message}")
        lower = max(0.0, min(1.0, float(lower_result.fun)))
        upper = max(0.0, min(1.0, float(-upper_result.fun)))
        if upper + 1e-9 < lower:
            raise RuntimeError(f"Inconsistent LP bounds for {rule!r}: {lower} > {upper}")
        product, marginals = _marginal_product(predicates, event_vectors, reference)
        lower_lift, upper_lift, status = _lift_bounds(lower, upper, product, marginals)
        reference_exposure = float(row @ reference)
        reference_lift = None
        if (status in ("defined", "zero_lower_exposure") and reference_exposure > 0.0
                and all(value > 0.0 for value in marginals)):
            reference_lift = math.log(reference_exposure) - math.fsum(
                math.log(value) for value in marginals)
        width = (upper_lift - lower_lift
                 if lower_lift is not None and upper_lift is not None else None)
        rows.append({
            "rule": rule,
            "status": "defined" if status == "defined" else status,
            "reference_exposure": reference_exposure,
            "lower_exposure": lower,
            "upper_exposure": upper,
            "marginal_product": product,
            "predicate_marginals": marginals,
            "reference_lift": reference_lift,
            "lower_lift": lower_lift,
            "upper_lift": upper_lift,
            "lift_width": width,
            "positive_lift": bool(upper_lift is not None and upper_lift > 1e-9),
        })

    defined = [entry for entry in rows if entry.get("lift_width") is not None]
    widths = [entry["lift_width"] for entry in defined]
    lower_lifts = [entry["lower_lift"] for entry in defined]
    upper_lifts = [entry["upper_lift"] for entry in rows if entry.get("upper_lift") is not None]
    positive = [entry for entry in rows if entry.get("positive_lift")]
    summary = {
        "rule_count": len(rows),
        "defined_rule_count": len(defined),
        "positive_lift_count": len(positive),
        "positive_lift_rules": [entry["rule"] for entry in positive],
        "min_lower_lift": min(lower_lifts) if lower_lifts else None,
        "max_upper_lift": max(upper_lifts) if upper_lifts else None,
        "median_lift_width": float(np.median(widths)) if widths else None,
        "mean_lift_width": float(np.mean(widths)) if widths else None,
        "max_lift_width": max(widths) if widths else None,
        "median_reference_lift": float(np.median(
            [entry["reference_lift"] for entry in defined])) if defined else None,
    }

    corner = None
    if positive:
        best = max(positive, key=lambda entry: entry["upper_lift"])
        row = _solver_row(report, best["rule"], names)
        result = linprog(-row, A_eq=matrix, b_eq=target, bounds=bounds, method="highs")
        if not result.success:
            raise RuntimeError(f"Linear solver failed for positive corner: {result.message}")
        corner_weights = {name: float(value) for name, value in zip(names, result.x,
                                                                    strict=True)}
        corner = {
            "rule": best["rule"],
            "lift": best["upper_lift"],
            "weights": corner_weights,
            "max_weight": max(corner_weights.values()),
            "floor_count": sum(value <= min_weight + 1e-8 for value in corner_weights.values()),
            "support_count": sum(value >= 1e-6 for value in corner_weights.values()),
        }

    return {
        "schema_version": 1,
        "source_seed": report.get("seed"),
        "samples_per_condition": report.get("samples_per_condition"),
        "min_weight": min_weight,
        "reference_weights": {name: float(value) for name, value in zip(names, reference,
                                                                        strict=True)},
        "summary": summary,
        "positive_corner": corner,
        "rules": rows,
        "note": ("Exact linear-program bounds on the estimated Monte Carlo matrix. "
                 "Singleton exposures are fixed at the reference mixture. Uncertainty "
                 "in the matrix is not propagated."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute exact LP bounds on rule exposure and composition lift")
    parser.add_argument("--report", type=Path, required=True,
                        help="subset-enabled measurement JSON")
    parser.add_argument("--weights", type=Path, required=True,
                        help="reference condition-weight JSON")
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/design-bounds.json"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"),
                            object_pairs_hook=unique_object)
        weights = json.loads(args.weights.read_text(encoding="utf-8"),
                             object_pairs_hook=unique_object)
        result = feasible_lift_bounds(report, weights, min_weight=args.min_weight)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
