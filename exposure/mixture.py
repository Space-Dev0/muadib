"""Combine a measured exposure matrix under a specified sampling mixture.

These are plug-in estimates: uncertainty from the original Monte Carlo sample
is not propagated, and zero empirical counts are not treated as impossibility.
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

from exposure.measure import load_rules, unique_object


def _probability(measurement: object, trials: int) -> float:
    if not isinstance(measurement, dict):
        raise ValueError("Malformed measurement")
    count, value = measurement.get("successes"), measurement.get("probability")
    if (type(count) is not int or not 0 <= count <= trials or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not math.isclose(value, count / trials, rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError("Measurement count and probability are inconsistent")
    return float(value)


def _log_lift(joint: float, marginals: list[float]) -> float | None:
    """Return log(joint / product(marginals)), or None for empirical zeros."""
    if joint <= 0 or any(value <= 0 for value in marginals):
        return None
    return math.log(joint) - sum(math.log(value) for value in marginals)


def _lift_status(joint: float, marginals: list[float]) -> str:
    if any(value <= 0 for value in marginals):
        return "zero_marginal"
    if joint <= 0:
        return "zero_joint"
    return "defined"


def summarize(report: dict, weights: dict) -> dict:
    """Calculate rule and pair lift from a subset-enabled measurement report."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Expected a schema version 1 measurement report")
    rules = load_rules(report.get("rules"))
    names = list(rules)
    if report.get("include_subsets") is not True:
        raise ValueError(
            "Measurement report is old or incomplete: include_subsets is not true. "
            "Rerun `python -m exposure.measure` with the original rule set and "
            "`--include-subsets`, choose a new --output path so the existing report is "
            "not overwritten, and pass that subset-enabled report to --report."
        )
    if not isinstance(weights, dict) or set(weights) != set(names):
        raise ValueError("Weights must contain exactly one entry per condition")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
           for w in weights.values()):
        raise ValueError("Weights must be finite nonnegative numbers")
    total = math.fsum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Weights must sum to 1")
    weights = {name: weights[name] / total for name in names}
    columns = report.get("columns")
    if not isinstance(columns, list) or len(columns) != len(names):
        raise ValueError("Expected one column for each condition")
    by_name = {}
    for column in columns:
        if not isinstance(column, dict) or column.get("condition") not in rules or column["condition"] in by_name:
            raise ValueError("Invalid or duplicate condition column")
        condition = column["condition"]
        trials = column.get("accepted_scenes")
        if type(trials) is not int or trials < 1:
            raise ValueError("Each column needs a positive number of accepted scenes")
        measurements, subsets = column.get("measurements"), column.get("predicate_subsets")
        if not isinstance(measurements, dict) or set(measurements) != set(names):
            raise ValueError("Each column must measure every rule")
        if not isinstance(subsets, dict) or set(subsets) != set(names):
            raise ValueError("Each column must measure each rule's subsets")
        parsed_subsets = {}
        for name, predicates in rules.items():
            expected = {(i,) for i in range(len(predicates))} | set(combinations(range(len(predicates)), 2))
            entries = subsets[name]
            if not isinstance(entries, list):
                raise ValueError("Malformed subset measurements")
            found = {}
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("predicate_indices"), list):
                    raise ValueError("Malformed subset indices")
                indices = tuple(entry["predicate_indices"])
                if any(type(i) is not int for i in indices) or indices not in expected or indices in found:
                    raise ValueError("Invalid or duplicate subset indices")
                found[indices] = _probability(entry, trials)
            if set(found) != expected:
                raise ValueError("Incomplete subset measurements")
            parsed_subsets[name] = found
        parsed_measurements = {name: _probability(measurements[name], trials) for name in names}
        if (parsed_measurements[condition] != 1.0 or
                any(value != 1.0 for value in parsed_subsets[condition].values())):
            raise ValueError("Conditioned scenes must satisfy their condition and its subsets")
        by_name[condition] = (parsed_measurements, parsed_subsets)
    results = {}
    for name, predicates in rules.items():
        joint = math.fsum(weights[c] * by_name[c][0][name] for c in names)
        marginal = [math.fsum(weights[c] * by_name[c][1][name][(i,)] for c in names)
                    for i in range(len(predicates))]
        pairs = []
        for i, j in combinations(range(len(predicates)), 2):
            pair_joint = math.fsum(weights[c] * by_name[c][1][name][(i, j)] for c in names)
            pairs.append({"predicate_indices": [i, j], "exposure": pair_joint,
                          "log_lift": _log_lift(pair_joint, [marginal[i], marginal[j]]),
                          "lift_status": _lift_status(pair_joint, [marginal[i], marginal[j]])})
        results[name] = {"rule_exposure": joint, "predicate_marginals": marginal,
                         "log_lift": _log_lift(joint, marginal),
                         "lift_status": _lift_status(joint, marginal), "pairs": pairs}
    return {"schema_version": 1, "source_seed": report.get("seed"),
            "samples_per_condition": report.get("samples_per_condition"),
            "weights": {name: weights[name] for name in names}, "rules": report["rules"],
            "results": results,
            "note": "Monte Carlo plug-in estimates; lift_status distinguishes empirical zeros. No uncertainty propagated."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exposures and lifts under a sampling mixture")
    parser.add_argument("--report", type=Path, required=True, help="subset-enabled measurement JSON")
    parser.add_argument("--weights", type=Path, required=True, help="JSON object mapping conditions to weights")
    parser.add_argument("--output", type=Path, default=Path("outputs/mixture.json"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        weights = json.loads(args.weights.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = summarize(report, weights)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
