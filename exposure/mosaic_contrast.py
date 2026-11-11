"""Descriptive run-level summary for a prospective MOSAIC-inspired contrast.

This analysis does not train models or decide whether the Stage 0 gate passed.
Each input row must describe a distinct trained model and its evaluated images.
"""

import argparse
import json
import math
from pathlib import Path


CELLS = (("low", "balanced"), ("low", "imbalanced"),
         ("high", "balanced"), ("high", "imbalanced"))


def summarize(data: object) -> dict:
    """Report equally weighted per-model accuracy and paired-seed contrasts."""
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list) or not data["runs"]:
        raise ValueError("Input must contain a nonempty runs list")
    by_seed: dict[int, dict[tuple[str, str], dict]] = {}
    run_ids: set[str] = set()
    for row in data["runs"]:
        if not isinstance(row, dict):
            raise ValueError("Every run must be an object")
        seed, run_id = row.get("seed"), row.get("run_id")
        successes, total = row.get("successes"), row.get("total")
        cell = (row.get("complexity"), row.get("imbalance"))
        if type(seed) is not int or seed < 0:
            raise ValueError("Run seed must be a nonnegative integer")
        if not isinstance(run_id, str) or not run_id.strip() or run_id in run_ids:
            raise ValueError("Each run needs a unique nonempty run_id")
        if cell not in CELLS:
            raise ValueError("Each run needs a recognized complexity and imbalance cell")
        if type(total) is not int or total <= 0 or type(successes) is not int or not 0 <= successes <= total:
            raise ValueError("Counts must be integers with 0 <= successes <= total and total > 0")
        runs = by_seed.setdefault(seed, {})
        if cell in runs:
            raise ValueError(f"Duplicate cell {cell} for seed {seed}")
        run_ids.add(run_id)
        runs[cell] = row

    results = []
    for seed, runs in sorted(by_seed.items()):
        if set(runs) != set(CELLS):
            raise ValueError(f"Seed {seed} must have exactly one run in every 2x2 cell")
        p = {cell: runs[cell]["successes"] / runs[cell]["total"] for cell in CELLS}
        complexity_drop = ((p[("low", "balanced")] - p[("high", "balanced")]) +
                           (p[("low", "imbalanced")] - p[("high", "imbalanced")])) / 2
        imbalance_drop = ((p[("low", "balanced")] - p[("low", "imbalanced")]) +
                          (p[("high", "balanced")] - p[("high", "imbalanced")])) / 2
        results.append({"seed": seed, "runs": [dict(runs[cell]) for cell in CELLS],
                        "complexity_drop": complexity_drop, "imbalance_drop": imbalance_drop,
                        "difference_in_drops": complexity_drop - imbalance_drop})
    metrics = ("complexity_drop", "imbalance_drop", "difference_in_drops")
    means = {key: math.fsum(row[key] for row in results) / len(results) for key in metrics}
    # A seed block, not an image, is the unit of training replication.
    variation = {}
    for key in metrics:
        if len(results) < 2:
            variation[key] = {"sample_sd": None, "standard_error": None}
        else:
            variance = math.fsum((row[key] - means[key]) ** 2 for row in results) / (len(results) - 1)
            sd = math.sqrt(variance)
            variation[key] = {"sample_sd": sd, "standard_error": sd / math.sqrt(len(results))}
    return {"schema_version": 1, "seed_count": len(results), "seeds": results,
            "mean": means, "between_seed_variation": variation,
            "note": ("Descriptive equal-cell-weighted run-level contrasts only. A positive "
                     "difference in drops is not by itself a replication or Stage 0 gate pass. "
                     "Standard errors assume independent seed blocks; no confidence interval is provided. "
                     "Comparability, independent model runs, checker validity, uncertainty, "
                     "the adapted protocol and the success criterion must be established separately.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON containing a runs list")
    parser.add_argument("--output", type=Path, help="new JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        report = summarize(json.loads(args.input.read_text(encoding="utf-8")))
        content = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(content, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(content)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
