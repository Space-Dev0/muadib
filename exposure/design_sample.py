"""Sample exploratory simplex designs preserving estimated singleton exposure.

Hit-and-run runs in the equality nullspace only; it does not enforce exposure
inequalities or establish mixing, independence, or Stage 0 criteria.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from exposure.design_subspace import diagnose
from exposure.measure import unique_object
from exposure.mixture import summarize


def sample_designs(report: dict, weights: dict, *, samples: int, seed: int,
                   min_weight: float = 0.0, burn_in: int = 100, thin: int = 10) -> dict:
    """Run seeded hit-and-run inside the floor-constrained equality polytope.

    Requires a strictly interior reference relative to the requested floor.
    Points are neither clipped nor renormalized; failed numerical checks abort.
    """
    for name, value, minimum in (("samples", samples, 1), ("burn_in", burn_in, 0), ("thin", thin, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(min_weight, bool) or not isinstance(min_weight, (int, float))
            or not math.isfinite(min_weight) or min_weight < 0):
        raise ValueError("min_weight must be a finite nonnegative number")
    summarize(report, weights)
    subspace = diagnose(report, weights)
    names = subspace["column_names"]
    if subspace["nullity"] == 0:
        raise ValueError("No singleton-preserving directions: nullity is zero")
    position = np.asarray([weights[name] for name in names], dtype=float)
    if abs(math.fsum(float(w) for w in position) - 1.0) > 1e-12:
        raise ValueError("Reference weights must sum to one within 1e-12; they are not renormalized")
    if min_weight * len(names) >= 1 or np.any(position <= min_weight + 1e-12):
        raise ValueError("Reference must be strictly inside the simplex above the minimum-weight floor")

    matrix = np.asarray([subspace["simplex_row"]] +
                        [event["column_probabilities"] for event in subspace["singleton_events"]])
    target = matrix @ position
    basis = np.asarray(subspace["nullspace_basis"], dtype=float)
    rng = np.random.default_rng(seed)
    designs = []
    total_steps = burn_in + samples * thin
    for step in range(total_steps):
        direction = rng.normal(size=subspace["nullity"]) @ basis
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= 1e-14:
            raise RuntimeError("Degenerate nullspace direction")
        direction /= norm
        low, high = -math.inf, math.inf
        for weight, delta in zip(position, direction, strict=True):
            if delta > 0:
                low = max(low, (min_weight - weight) / delta)
            elif delta < 0:
                high = min(high, (min_weight - weight) / delta)
        if not math.isfinite(low) or not math.isfinite(high) or high - low <= 1e-12:
            raise RuntimeError("Degenerate or unbounded hit-and-run step interval")
        position = position + float(rng.uniform(low, high)) * direction
        if (not np.all(np.isfinite(position)) or np.any(position <= min_weight)
                or abs(math.fsum(float(w) for w in position) - 1.0) > 1e-7
                or float(np.max(np.abs(matrix @ position - target))) > 1e-7):
            raise RuntimeError("Sample reached a boundary or violated a numerical constraint")
        if step >= burn_in and (step - burn_in + 1) % thin == 0:
            designs.append({"name": f"sample_{len(designs) + 1:04d}",
                            "weights": {name: float(value) for name, value in zip(names, position, strict=True)}})
    return {"designs": designs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample singleton-preserving exploratory designs")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/design-samples.json"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        weights = json.loads(args.weights.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = sample_designs(report, weights, samples=args.samples, seed=args.seed,
                                min_weight=args.min_weight, burn_in=args.burn_in, thin=args.thin)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
