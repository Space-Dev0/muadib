"""Find sampling weights satisfying exposure bounds on an estimated matrix."""

import argparse
import json
import math
from pathlib import Path

from scipy.optimize import linprog

from exposure.measure import load_rules, unique_object
from exposure.mixture import summarize


def solve(report: dict, specification: dict, *, min_weight: float = 0.0) -> dict:
    """Solve feasibility on the supplied Monte Carlo matrix, not the true matrix."""
    if (isinstance(min_weight, bool) or not isinstance(min_weight, (int, float))
            or not math.isfinite(min_weight) or min_weight < 0):
        raise ValueError("min_weight must be a finite nonnegative number")
    if not isinstance(report, dict):
        raise ValueError("Measurement report must be a JSON object")
    rules = load_rules(report.get("rules"))
    names = list(rules)
    summarize(report, {name: 1 / len(names) for name in names})
    if not isinstance(specification, dict) or set(specification) != {"constraints"}:
        raise ValueError("Specification must contain only a constraints list")
    constraints = specification["constraints"]
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("Constraints must be a nonempty list")
    columns = {column["condition"]: column for column in report["columns"]}
    a_ub, b_ub, rows, labels = [], [], [], set()
    for entry in constraints:
        if not isinstance(entry, dict) or not {"name", "rule", "kind", "predicate_indices"} <= entry.keys():
            raise ValueError("Each constraint requires name, rule, kind, and predicate_indices")
        if set(entry) - {"name", "rule", "kind", "predicate_indices", "min", "max"}:
            raise ValueError("Unexpected constraint field")
        label, rule, kind, indices = (entry[k] for k in ("name", "rule", "kind", "predicate_indices"))
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValueError("Constraint names must be unique nonempty strings")
        labels.add(label)
        if not isinstance(rule, str) or rule not in rules or kind not in ("rule", "predicate", "pair"):
            raise ValueError(f"Invalid rule or kind for {label}")
        size = {"rule": 0, "predicate": 1, "pair": 2}[kind]
        if (not isinstance(indices, list) or len(indices) != size or
                any(type(i) is not int or not 0 <= i < len(rules[rule]) for i in indices) or
                indices != sorted(set(indices))):
            raise ValueError(f"Invalid predicate indices for {label}")
        lower, upper = entry.get("min"), entry.get("max")
        if any(key in entry and entry[key] is None for key in ("min", "max")):
            raise ValueError(f"Bounds cannot be null for {label}")
        if lower is None and upper is None:
            raise ValueError(f"At least one bound is required for {label}")
        for bound in (lower, upper):
            if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or
                                      not math.isfinite(bound) or not 0 <= bound <= 1):
                raise ValueError(f"Bounds must be finite numbers in [0, 1] for {label}")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Lower bound exceeds upper bound for {label}")
        row = []
        for name in names:
            column = columns[name]
            if kind == "rule":
                probability = column["measurements"][rule]["probability"]
            else:
                probability = next(item["probability"] for item in column["predicate_subsets"][rule]
                                   if item["predicate_indices"] == indices)
            row.append(float(probability))
        rows.append(row)
        if lower is not None:
            a_ub.append([-v for v in row])
            b_ub.append(-lower)
        if upper is not None:
            a_ub.append(row)
            b_ub.append(upper)

    solution = linprog([0.0] * len(names), A_ub=a_ub, b_ub=b_ub,
                       A_eq=[[1.0] * len(names)], b_eq=[1.0],
                       bounds=[(min_weight, None)] * len(names), method="highs")
    result = {"schema_version": 1, "source_seed": report.get("seed"),
              "samples_per_condition": report.get("samples_per_condition"),
              "rules": report["rules"], "specification": specification,
              "min_weight": min_weight,
              "note": "Feasibility applies to the estimated Monte Carlo matrix; uncertainty is not propagated."}
    if solution.status == 2:
        return {**result, "status": "infeasible", "solver_message": solution.message}
    if not solution.success:
        raise RuntimeError(f"Linear solver failed (status {solution.status}): {solution.message}")
    weights = {name: float(value) for name, value in zip(names, solution.x, strict=True)}
    if any(not math.isfinite(value) for value in weights.values()):
        raise RuntimeError("Solver returned non-finite weights")
    lowest_weight = min(weights.values())
    violations = [abs(math.fsum(weights.values()) - 1), max(0.0, -lowest_weight),
                  max(0.0, min_weight - lowest_weight)]
    achieved = []
    for entry, row in zip(constraints, rows, strict=True):
        value = math.fsum(weights[name] * v for name, v in zip(names, row, strict=True))
        lower = value - entry["min"] if "min" in entry and entry["min"] is not None else None
        upper = entry["max"] - value if "max" in entry and entry["max"] is not None else None
        violations.extend(max(0.0, -slack) for slack in (lower, upper) if slack is not None)
        achieved.append({"name": entry["name"], "exposure": value,
                         "lower_slack": lower, "upper_slack": upper})
    max_violation = max(violations)
    if max_violation > 1e-7:
        raise RuntimeError(f"Solver returned a solution with constraint violation {max_violation}")
    return {**result, "status": "feasible", "weights": weights,
            "constraints": achieved, "max_violation": max_violation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check exposure design feasibility")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--min-weight", type=float, default=0.0,
                        help="minimum weight per conditioning template")
    parser.add_argument("--output", type=Path, default=Path("outputs/design.json"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        specification = json.loads(args.constraints.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = solve(report, specification, min_weight=args.min_weight)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output} ({result['status']})")


if __name__ == "__main__":
    main()
