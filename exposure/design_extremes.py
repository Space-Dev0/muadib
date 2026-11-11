"""Construct exploratory singleton-preserving LP endpoint designs.

Endpoints maximize or minimize one full-rule exposure at a time. They use a
Monte Carlo matrix and can overfit it; evaluate on an independent matrix.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from exposure.design_subspace import diagnose
from exposure.measure import unique_object


def generate(report: dict, reference: dict, *, min_weight: float = 0.0) -> dict:
    """Return a diagnostics-compatible configuration with one endpoint per rule and direction."""
    if (isinstance(min_weight, bool) or not isinstance(min_weight, (float, int))
            or not math.isfinite(min_weight) or min_weight < 0):
        raise ValueError("min_weight must be finite and nonnegative")
    subspace = diagnose(report, reference)
    names = subspace["column_names"]
    base = np.array([subspace["reference_weights"][name] for name in names])
    if np.any(base < min_weight - 1e-12):
        raise ValueError("Reference weights must satisfy the minimum-weight floor")
    rows = [subspace["simplex_row"]] + [item["column_probabilities"] for item in subspace["singleton_events"]]
    matrix = np.asarray(rows, dtype=float)
    rhs = matrix @ base
    columns = {c["condition"]: c for c in report["columns"]}
    designs = [{"name": "reference", "weights": dict(zip(names, map(float, base), strict=True))}]
    for index, rule in enumerate(names):
        objective = np.array([columns[n]["measurements"][rule]["probability"] for n in names], dtype=float)
        for direction, sign in (("min", 1), ("max", -1)):
            solution = linprog(sign * objective, A_eq=matrix, b_eq=rhs,
                               bounds=[(min_weight, None)] * len(names), method="highs")
            if not solution.success:
                raise RuntimeError(f"LP failed for {rule} ({direction}): {solution.message}")
            values = solution.x
            if (not np.all(np.isfinite(values)) or np.max(np.abs(matrix @ values - rhs)) > 1e-7
                    or np.min(values) < min_weight - 1e-7):
                raise RuntimeError("LP returned a solution violating its equality or floor")
            designs.append({"name": f"endpoint_{index:03d}_{direction}",
                            "weights": dict(zip(names, map(float, values), strict=True))})
    return {"designs": designs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = [json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
                for path in (args.report, args.reference)]
        result = generate(*data, min_weight=args.min_weight)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
