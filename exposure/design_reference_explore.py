"""Explore reference mixtures; this is not a proposal gate evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from exposure.measure import load_rules, unique_object
from exposure.rules import Relation


def matrix(report):
    """Return mean singleton exposure per rule and condition and relation counts."""
    rules = load_rules(report["rules"])
    names = list(rules)
    columns = {column["condition"]: column for column in report["columns"]}
    if set(columns) != set(names) or len(columns) != len(report["columns"]):
        raise ValueError("Expected exactly one column per rule")
    rows = []
    counts = []
    for name, predicates in rules.items():
        rows.append([
            np.mean([
                next(cell["probability"] for cell in columns[condition]["predicate_subsets"][name]
                     if cell["predicate_indices"] == [index])
                for index in range(len(predicates))
            ]) for condition in names
        ])
        counts.append(sum(isinstance(predicate, Relation) for predicate in predicates))
    values = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Invalid singleton probabilities")
    return names, values, np.asarray(counts, dtype=float)


def correlation(values, counts, weights):
    marginal = values @ weights
    if np.any(marginal <= 0):
        raise ValueError("Nonpositive mean marginal exposure")
    x = np.log(marginal)
    if np.std(x) < 1e-14 or np.std(counts) < 1e-14:
        raise ValueError("Correlation is undefined for a constant predictor")
    return float(np.corrcoef(x, counts)[0, 1])


def evaluate(values, counts, weights):
    r = correlation(values, counts, weights)
    return {"correlation": r, "pairwise_vif_lower_bound": 1 / (1 - r * r) if abs(r) < 1 else None}


def optimize_reference(values, counts, minimum=0.001, maximum=0.04):
    n = values.shape[1]
    if not (0 <= minimum <= maximum <= 1 and n * minimum <= 1 <= n * maximum):
        raise ValueError("Infeasible weight bounds")
    uniform = np.full(n, 1 / n)
    result = minimize(lambda w: correlation(values, counts, w) ** 2, uniform,
                      method="SLSQP", bounds=[(minimum, maximum)] * n,
                      constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
                      options={"maxiter": 1000, "ftol": 1e-12})
    w = result.x
    if (not result.success or abs(w.sum() - 1) > 1e-7
            or np.any(w < minimum - 1e-7) or np.any(w > maximum + 1e-7)):
        raise ValueError(f"Reference optimization failed: {result.message}")
    return w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--holdout-report", type=Path, required=True)
    parser.add_argument("--min-weight", type=float, default=0.001)
    parser.add_argument("--max-weight", type=float, default=0.04)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        train = json.loads(args.training_report.read_text(), object_pairs_hook=unique_object)
        holdout = json.loads(args.holdout_report.read_text(), object_pairs_hook=unique_object)
        names, values, counts = matrix(train)
        holdout_names, holdout_values, holdout_counts = matrix(holdout)
        if names != holdout_names or not np.array_equal(counts, holdout_counts) or train["rules"] != holdout["rules"]:
            raise ValueError("Training and holdout rule definitions/order do not match")
        w = optimize_reference(values, counts, args.min_weight, args.max_weight)
        uniform = np.full(len(names), 1 / len(names))
        result = {
            "interpretation": "Exploratory pairwise bound only; not the full proposal gate.",
            "training_report": str(args.training_report), "holdout_report": str(args.holdout_report),
            "bounds": {"minimum": args.min_weight, "maximum": args.max_weight},
            "weights": dict(zip(names, map(float, w))),
            "training_uniform": evaluate(values, counts, uniform),
            "training_fitted": evaluate(values, counts, w),
            "holdout_uniform": evaluate(holdout_values, holdout_counts, uniform),
            "holdout_frozen": evaluate(holdout_values, holdout_counts, w),
            "caveats": ["Changing the reference mixture changes the exposure target.",
                        "This pairwise bound does not establish the full VIF or correlation gate.",
                        "Singleton-preserving designs still have no within-rule marginal variation.",
                        "SLSQP gives a local numerical solution, not a global optimum."],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, KeyError, ValueError, StopIteration, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
