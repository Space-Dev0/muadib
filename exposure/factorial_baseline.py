"""Measure an exploratory factorial subset on unconditioned annotated scenes."""

import argparse
import json
import random
from pathlib import Path

from exposure.factorial_relations import build
from exposure.measure import load_rules, wilson_interval
from exposure.render import _sample_objects
from exposure.rules import Relation, satisfies


def estimate(samples: int, seed: int) -> dict:
    """Count full rules and independently rebound relation events on shared scenes."""
    if type(samples) is not int or samples < 1 or type(seed) is not int:
        raise ValueError("samples must be a positive integer and seed an integer")
    rules = load_rules(build())
    relations = {kind: (Relation("a", kind, "b"),) for kind in ("left_of", "above")}
    counts = {name: 0 for name in rules}
    singleton = {name: 0 for name in relations}
    rng = random.Random(seed)
    for _ in range(samples):
        scene = _sample_objects(rng)
        for name, predicates in rules.items():
            counts[name] += satisfies(scene, predicates)
        for name, predicates in relations.items():
            singleton[name] += satisfies(scene, predicates)

    def metrics(n: int) -> dict:
        return {"successes": n, "probability": n / samples,
                "wilson_95": list(wilson_interval(n, samples))}

    return {"schema_version": 1, "seed": seed, "samples": samples,
            "rules": {name: metrics(n) for name, n in counts.items()},
            "relation_singletons": {name: metrics(n) for name, n in singleton.items()},
            "note": ("Unconditioned synthetic scenes only. Events share scenes; intervals are "
                     "pointwise, not simultaneous. This is not a Stage 0 gate or model result.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = estimate(args.samples, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
