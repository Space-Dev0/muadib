"""Describe two abstaining checkers with paired seed-block resampling.

Intended for exploratory data only: resampling seed blocks does not undo
adaptive detector development, uncertain human labels, or coverage gaps.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from exposure.measure import unique_object


def compare(data: object, *, bootstrap: int = 2000, seed: int = 81217) -> dict:
    """Compare fixed binary-or-abstain decisions on the same labeled items."""
    if type(bootstrap) is not int or bootstrap < 1 or type(seed) is not int:
        raise ValueError("bootstrap must be a positive integer and seed an integer")
    if not isinstance(data, dict) or not isinstance(data.get("records"), list) or not data["records"]:
        raise ValueError("A nonempty records list is required")
    blocks = defaultdict(dict)
    for row in data["records"]:
        if not isinstance(row, dict) or not {"seed", "condition", "human", "standard", "candidate"} <= row.keys():
            raise ValueError("Each record requires seed, condition, human and both predictions")
        block, condition = row["seed"], row["condition"]
        if type(block) is not int or block < 0 or not isinstance(condition, str) or not condition.strip():
            raise ValueError("Seed and condition must identify a valid block")
        if condition in blocks[block]:
            raise ValueError("Duplicate seed-by-condition record")
        if type(row["human"]) is not bool or any(value is not None and type(value) is not bool
                                                  for value in (row["standard"], row["candidate"])):
            raise ValueError("Human labels must be boolean and predictions boolean or null")
        blocks[block][condition] = row
    expected = set(next(iter(blocks.values())))
    if any(set(rows) != expected for rows in blocks.values()):
        raise ValueError("Each seed block must contain the same conditions")

    flat = [row for group in blocks.values() for row in group.values()]
    def counts(key):
        accepted = [row for row in flat if row[key] is not None]
        return {"accepted": len(accepted), "abstained": len(flat) - len(accepted),
                "accepted_disagreements": sum(row[key] != row["human"] for row in accepted),
                "coverage": len(accepted) / len(flat)}
    standard, candidate = counts("standard"), counts("candidate")
    keys = sorted(blocks)
    contributions = {key: sum((row["candidate"] is not None) - (row["standard"] is not None)
                              for row in blocks[key].values()) / len(expected) for key in keys}
    difference = candidate["coverage"] - standard["coverage"]
    interval = None
    if len(keys) >= 2:
        rng = random.Random(seed)
        draws = [sum(contributions[rng.choice(keys)] for _ in keys) / len(keys)
                 for _ in range(bootstrap)]
        interval = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
    return {"schema_version": 1, "images": len(flat), "seed_blocks": len(keys),
            "conditions": sorted(expected), "standard": standard, "candidate": candidate,
            "coverage_difference": difference, "paired_seed_bootstrap_95": interval,
            "bootstrap_draws": bootstrap, "bootstrap_seed": seed,
            "note": ("Descriptive coverage comparison only. Bootstrap treats seed blocks as "
                     "exchangeable and ignores human-label uncertainty, adaptive tuning and other "
                     "dependence. Accepted disagreements exclude abstentions. Not independent "
                     "validation, a confidence interval for error, or a Stage 0 gate.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=81217)
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = compare(data, bootstrap=args.bootstrap, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
