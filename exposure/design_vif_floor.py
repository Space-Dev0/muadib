"""Bound fixed-marginal design VIF from a fixed between-rule correlation.

This is a diagnostic for the supplied rows and assumptions, not a Stage 0 gate.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exposure.design_strata import summarize
from exposure.measure import unique_object


def bound(report: dict) -> dict:
    """Analyze complete rule-by-design rows with constant per-rule marginals.

    In a regression with an intercept and both marginal exposure and relation
    count as predictors, the VIF of either is at least 1/(1-r**2): adding more
    predictors cannot reduce the R-squared of the auxiliary regression.
    """
    summarize(report)  # Validate rows, uniqueness and finite values.
    grouped = defaultdict(dict)
    for row in report["rows"]:
        grouped[row["rule"]][row["design"]] = row
    design_sets = [set(rows) for rows in grouped.values()]
    if any(s != design_sets[0] for s in design_sets[1:]):
        raise ValueError("Every rule must have the same design set")
    values = []
    max_range = 0.0
    for rows in grouped.values():
        marginals = [row["log_mean_marginal"] for row in rows.values()]
        spread = max(marginals) - min(marginals)
        max_range = max(max_range, spread)
        if spread > 1e-12:
            raise ValueError("Marginal exposure varies within a rule")
        values.append((sum(marginals) / len(marginals), next(iter(rows.values()))["k_rel"]))
    matrix = np.asarray(values, dtype=float)
    if len(matrix) < 3 or np.any(np.std(matrix, axis=0) <= 1e-12):
        raise ValueError("Both predictors need variation across at least three rules")
    correlation = float(np.clip(np.corrcoef(matrix.T)[0, 1], -1, 1))
    denominator = 1 - correlation * correlation
    return {"schema_version": 1, "rules": len(grouped), "designs": len(design_sets[0]),
            "max_within_rule_marginal_range": max_range, "correlation": correlation,
            "vif_lower_bound": 1 / denominator if denominator > 1e-12 else None,
            "status": "finite" if denominator > 1e-12 else "perfect_collinearity",
            "note": ("Bound applies only when these two predictors and an intercept are included, "
                     "rows are complete and equally weighted by rule, and singleton marginals are "
                     "fixed on this estimated matrix. Not a population or Stage 0 gate result.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = bound(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
