"""Measure base-renderer support for exploratory sector-bound endpoint pairs.

This is a sampling-cost diagnostic, not a benchmark or generated-image audit.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import random

from exposure.measure import wilson_interval
from exposure.render import COLORS, SHAPES, _sample_objects
from exposure.spatial_sector import sector_holds


def measure(samples: int, seed: int) -> dict:
    """Count scene-level existential events, once per event per scene."""
    if type(samples) is not int or samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("samples must be positive and seed nonnegative integers")
    atoms = tuple(product(COLORS, SHAPES))
    keys = tuple(product(("left", "above"), atoms, atoms))
    counts = dict.fromkeys(keys, 0)
    rng = random.Random(seed)
    for _ in range(samples):
        objects = _sample_objects(rng)
        hits = set()
        for i, subject in enumerate(objects):
            for j, target in enumerate(objects):
                if i == j:
                    continue
                for direction in ("left", "above"):
                    if sector_holds(subject, target, direction=direction):
                        hits.add((direction, (subject.color, subject.shape),
                                  (target.color, target.shape)))
        for key in hits:
            counts[key] += 1
    rows = []
    for (direction, subject, target), count in counts.items():
        rows.append({"direction": direction, "subject": list(subject),
                     "target": list(target), "count": count,
                     "probability": count / samples,
                     "pointwise_wilson_95": list(wilson_interval(count, samples))})
    return {"schema_version": 1, "samples": samples, "seed": seed,
            "sector": {"half_angle_degrees": 15, "min_distance": 14, "max_distance": 60},
            "rows": rows,
            "note": "Exploratory base-renderer counts, not benchmark or checker validation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=82811)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.samples, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    counts = [row["count"] for row in result["rows"]]
    print(f"{len(counts)} events; count range {min(counts)}–{max(counts)}; "
          f"{sum(count == 0 for count in counts)} unobserved")


if __name__ == "__main__":
    main()
