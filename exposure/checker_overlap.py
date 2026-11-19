"""Paired synthetic occlusion stress test for existing rule checkers.

Renderer annotations are not human visual labels, especially after overlap.
This is an exploratory instrument diagnostic, not a generated-image audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from exposure.checker import DetectionError, detect, evaluate_variable_count_experimental
from exposure.checker_calibration import _objects, _rule
from exposure.render import COLORS, SHAPES, _draw, parse_rule
from exposure.rules import SceneObject, satisfies


def paired_scenes(seed: int, pairs: int):
    """Yield balanced positive/negative annotation pairs differing only in geometry."""
    if type(seed) is not int or seed < 0 or type(pairs) is not int or pairs < 2 or pairs % 2:
        raise ValueError("seed must be nonnegative and pairs must be positive and even")
    rng = random.Random(seed)
    seen = set()
    for index in range(pairs):
        scenario = "present_competing" if index % 2 == 0 else "swapped_binding"
        for _ in range(1000):
            color, shape = rng.choice(tuple(COLORS)), rng.choice(SHAPES)
            source = _objects(rng, scenario, color, shape)
            signature = tuple((obj.color, obj.shape) for obj in source)
            if signature not in seen:
                seen.add(signature)
                break
        else:
            raise ValueError("unable to construct enough unique attribute sequences")
        # The first two objects are separated by 20 pixels in the control and
        # overlap by 8 pixels (nominal 12-pixel shape width) in the treatment.
        # They retain their draw order, so the second may hide the first.
        control_positions = ((16, 20), (36, 20), (48, 46))
        overlap_positions = ((16, 20), (20, 20), (48, 46))
        def place(positions):
            return tuple(SceneObject(obj.color, obj.shape, *point)
                         for obj, point in zip(source, positions, strict=True))
        yield index, parse_rule(_rule(color, shape)), place(control_positions), place(overlap_positions)


def run(seed: int, pairs: int) -> dict:
    counts = defaultdict(Counter)
    details = []
    for index, rule, control, overlap in paired_scenes(seed, pairs):
        truth = satisfies(control, rule)
        if satisfies(overlap, rule) != truth:
            raise RuntimeError("attribute-only truth changed after geometry shift")
        decisions = {}
        for condition, objects in (("separated", control), ("overlap", overlap)):
            image = _draw(objects).image
            for method in ("standard", "experimental"):
                try:
                    result = (satisfies(detect(image), rule) if method == "standard"
                              else evaluate_variable_count_experimental(image, rule))
                except DetectionError:
                    result = None
                if result is not None and type(result) is not bool:
                    raise ValueError("invalid checker output")
                decision = "abstain" if result is None else "yes" if result else "no"
                group = f"{condition}:{method}"
                key = ("positive" if truth else "negative") + "_" + decision
                counts[group][key] += 1
                decisions[group] = decision
        details.append({"pair": index, "metadata_truth": truth, "decisions": decisions})
    categories = [f"{truth}_{answer}" for truth in ("positive", "negative")
                  for answer in ("yes", "no", "abstain")]
    return {"schema_version": 1, "seed": seed, "pairs": pairs,
            "counts": {group: {key: counter[key] for key in categories}
                       for group, counter in sorted(counts.items())},
            "details": details,
            "scope": "Paired synthetic metadata truth; overlap may hide objects and make labels visually ambiguous. "
                     "Not a human-labeled or generated-image validation and not a Stage 0 gate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=82995)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.seed, args.pairs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
