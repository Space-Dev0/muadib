"""Describe pooled and within-rule exposure correlations, not a research gate."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from exposure.measure import unique_object


def _correlation(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.any(np.std(values, axis=0) <= 1e-12):
        return None
    return float(np.clip(np.corrcoef(values.T)[0, 1], -1, 1))


def summarize(report):
    """Compare raw and rule-centered correlations on the same retained rows."""
    if not isinstance(report, dict) or not isinstance(report.get("rows"), list) or not report["rows"]:
        raise ValueError("A nonempty design diagnostics rows list is required")
    rows = report["rows"]
    seen = set()
    grouped = defaultdict(list)
    strata = defaultdict(list)
    rule_complexity = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each row must be an object")
        design, rule = row.get("design"), row.get("rule")
        if not all(isinstance(x, str) and x.strip() for x in (design, rule)):
            raise ValueError("Each row must name its design and rule")
        if (design, rule) in seen:
            raise ValueError("Duplicate rule-by-design row")
        seen.add((design, rule))
        try:
            pair = [row["log_mean_marginal"], row["log_lift"]]
            complexity = (row["k_attr"], row["k_rel"])
        except KeyError as exc:
            raise ValueError(f"Missing row field {exc}") from exc
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pair):
            raise ValueError("Exposure features must be finite numbers")
        if any(type(x) is not int or x < 0 for x in complexity):
            raise ValueError("Complexity counts must be nonnegative integers")
        if rule in rule_complexity and rule_complexity[rule] != complexity:
            raise ValueError("Rule complexity must be consistent across designs")
        rule_complexity[rule] = complexity
        grouped[rule].append(pair)
        strata[complexity].append((rule, pair))

    pairs = [pair for values in grouped.values() for pair in values]
    means = {rule: np.mean(values, axis=0) for rule, values in grouped.items()}
    centered = [np.asarray(pair) - means[rule] for rule, values in grouped.items() for pair in values]
    # Replicate each mean once per retained row, matching pooled row weighting.
    replicated_means = [means[rule] for rule, values in grouped.items() for _ in values]
    return {
        "schema_version": 1,
        "source_seed": report.get("source_seed"),
        "retained_rows": len(rows),
        "excluded_rows": report.get("excluded_count"),
        "rule_count": len(grouped),
        "design_count": len({row["design"] for row in rows}),
        "pooled_correlation": _correlation(pairs),
        "within_rule_correlation": _correlation(centered),
        "row_weighted_rule_mean_correlation": _correlation(replicated_means),
        "strata": [
            {"k_attr": key[0], "k_rel": key[1], "row_count": len(items),
             "rule_count": len({rule for rule, _ in items}),
             "correlation": _correlation([pair for _, pair in items])}
            for key, items in sorted(strata.items())
        ],
        "gate_status": "not_assessed",
        "note": ("Exploratory descriptive correlations, not a gate or causal decomposition. "
                 "Rows share designs and rules, exposure estimates are uncertain, and excluded "
                 "rows may bias comparisons. Rule-mean correlation weights each rule by its "
                 "retained row count. Breadth and uncertainty remain unassessed."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = summarize(report)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            out.write(content)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
