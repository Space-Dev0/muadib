"""Construct an exploratory, versioned, endpoint-bound sector scene manifest.

This is a new benchmark candidate and does not change historical rule semantics.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random

from exposure.render import COLORS, SHAPES, _draw, _sample_objects
from exposure.spatial_sector import exists_sector

SCHEMA = "bounded-sector-scenes-v1"
PARAMETERS = {"half_angle_degrees": 15, "min_distance": 14, "max_distance": 60}


def rule_inventory() -> list[dict]:
    """Return all ordered endpoint combinations, balanced across directions."""
    atoms = tuple(product(COLORS, SHAPES))
    return [{"id": f"{a[0]}_{a[1]}_{direction}_{b[0]}_{b[1]}",
             "direction": direction,
             "subject": {"color": a[0], "shape": a[1]},
             "target": {"color": b[0], "shape": b[1]}}
            for direction, a, b in product(("left", "above"), atoms, atoms)]


def generate(*, train_seed: int, heldout_seed: int, scenes_per_rule: int,
             max_attempts: int = 10000) -> dict:
    """Rejection-sample scenes with independently seeded streams and unique pixels."""
    if (any(type(value) is not int or value < 0 for value in (train_seed, heldout_seed))
            or train_seed == heldout_seed or type(scenes_per_rule) is not int
            or scenes_per_rule < 1 or type(max_attempts) is not int or max_attempts < 1):
        raise ValueError("distinct nonnegative seeds and positive integer budgets required")
    rules = rule_inventory()
    records = []
    pixels_seen: set[str] = set()
    for split, seed in (("train", train_seed), ("heldout", heldout_seed)):
        rng = random.Random(seed)
        for rule in rules:
            for index in range(scenes_per_rule):
                for attempt in range(1, max_attempts + 1):
                    objects = _sample_objects(rng)
                    if not exists_sector(objects, direction=rule["direction"],
                                         subject_filter=rule["subject"],
                                         target_filter=rule["target"], **PARAMETERS):
                        continue
                    pixel_hash = hashlib.sha256(_draw(objects).image.tobytes()).hexdigest()
                    if pixel_hash in pixels_seen:
                        continue
                    pixels_seen.add(pixel_hash)
                    records.append({"split": split, "rule_id": rule["id"], "index": index,
                                    "attempts": attempt, "pixel_sha256": pixel_hash,
                                    "objects": [vars(obj) for obj in objects]})
                    break
                else:
                    raise RuntimeError(f"Sampling budget exhausted for {split}/{rule['id']}/{index}")
    return {"schema": SCHEMA, "parameters": PARAMETERS, "rules": rules,
            "train_seed": train_seed, "heldout_seed": heldout_seed,
            "scenes_per_rule": scenes_per_rule, "max_attempts": max_attempts,
            "records": records,
            "note": "Exploratory positive conditioned scenes; not a validated benchmark or Stage 0 pass."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--heldout-seed", type=int, required=True)
    parser.add_argument("--scenes-per-rule", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = generate(train_seed=args.train_seed, heldout_seed=args.heldout_seed,
                          scenes_per_rule=args.scenes_per_rule, max_attempts=args.max_attempts)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(result['records'])} scenes for {len(result['rules'])} rules")


if __name__ == "__main__":
    main()
