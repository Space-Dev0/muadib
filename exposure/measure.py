"""Estimate scene-level co-satisfaction for a set of conditioned rules.

Rows are measured rules; columns are conditioning rules. These estimates
describe the prototype renderer, not training results or a fitted design.
"""

import argparse
import json
import math
import random
from itertools import combinations
from pathlib import Path

from exposure.render import parse_rule, sample_conditioned
from exposure.rules import Predicate, satisfies


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON keys, which would otherwise silently alter rules."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_rules(data: object) -> dict[str, tuple[Predicate, ...]]:
    """Parse an ordered JSON mapping of distinct names to predicate lists."""
    if not isinstance(data, dict) or not data:
        raise ValueError("Rule set must be a nonempty JSON object")
    result = {}
    for name, predicates in data.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Rule names must be nonempty strings")
        try:
            result[name] = parse_rule(predicates)
        except ValueError as exc:
            raise ValueError(f"Invalid rule {name!r}: {exc}") from exc
    return result


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Approximate 95% binomial Wilson interval, including at zero and one."""
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def estimate(rules: dict[str, tuple[Predicate, ...]], *, samples: int, seed: int,
             max_attempts: int = 10_000, include_subsets: bool = False) -> dict:
    """Sample scenes for each condition and count all satisfied rules.

    Each column uses `samples` accepted scenes. An exhausted rejection budget
    aborts the run; it is never counted as an accepted or negative observation.
    """
    if not rules or samples < 1 or max_attempts < 1:
        raise ValueError("Rules, samples, and max_attempts must be nonempty/positive")
    rng = random.Random(seed)
    subset_indices = {
        name: [(i,) for i in range(len(predicates))] + list(combinations(range(len(predicates)), 2))
        for name, predicates in rules.items()
    } if include_subsets else {}
    columns = []
    for condition, predicates in rules.items():
        counts = {name: 0 for name in rules}
        subset_counts = {name: [0] * len(indices) for name, indices in subset_indices.items()}
        attempts = 0
        for index in range(samples):
            try:
                objects, used = sample_conditioned(rng, predicates, max_attempts)
            except ValueError as exc:
                raise ValueError(f"Condition {condition!r}, sample {index + 1}: {exc}") from exc
            attempts += used
            for name, measured_rule in rules.items():
                counts[name] += satisfies(objects, measured_rule)
                if include_subsets:
                    for position, indices in enumerate(subset_indices[name]):
                        subset = tuple(measured_rule[i] for i in indices)
                        subset_counts[name][position] += satisfies(objects, subset)
        column = {
            "condition": condition,
            "accepted_scenes": samples,
            "total_attempts": attempts,
            "measurements": {
                name: {"successes": count, "probability": count / samples,
                       "wilson_95": ([1.0, 1.0] if name == condition else
                                     list(wilson_interval(count, samples)))}
                for name, count in counts.items()
            },
        }
        if include_subsets:
            column["predicate_subsets"] = {
                name: [
                    {"predicate_indices": list(indices), "successes": count,
                     "probability": count / samples,
                     "wilson_95": ([1.0, 1.0] if name == condition else
                                   list(wilson_interval(count, samples)))}
                    for indices, count in zip(subset_indices[name], subset_counts[name], strict=True)
                ]
                for name in rules
            }
        columns.append(column)
    return {"schema_version": 1, "seed": seed, "samples_per_condition": samples,
            "include_subsets": include_subsets,
            "max_attempts_per_scene": max_attempts,
            "interpretation": "scene satisfaction given requested condition",
            "rule_order": list(rules), "columns": columns}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate rule co-satisfaction across scene conditions")
    parser.add_argument("--rules", type=Path, required=True, help="JSON mapping rule names to predicates")
    parser.add_argument("--samples", type=int, default=1000, help="accepted scenes per condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=10_000, help="attempt budget per scene")
    parser.add_argument("--include-subsets", action="store_true",
                        help="also measure singleton and pair subsets within each rule")
    parser.add_argument("--output", type=Path, default=Path("outputs/exposure.json"))
    args = parser.parse_args()
    try:
        rule_data = json.loads(args.rules.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        rules = load_rules(rule_data)
        report = estimate(rules, samples=args.samples, seed=args.seed, max_attempts=args.max_attempts,
                          include_subsets=args.include_subsets)
        report["rules"] = rule_data
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
