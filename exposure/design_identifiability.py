"""Inspect fixed-rule-intercept identifiability of exploratory exposure features.

This is a structural fixed-effects diagnostic, not a GLMM fit or Stage 0 gate.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exposure.design_strata import summarize as validate_rows
from exposure.measure import unique_object

FEATURES = ("log_mean_marginal", "log_lift", "log_lift_x_k_rel")


def inspect(report: dict) -> dict:
    """Remove rule means and inspect rank of exposure and interaction columns.

    Columns are scaled to unit norm before SVD; zero-variation columns are
    reported separately. A coefficient is identified only when its coordinate
    is absent from the numerical null space of the full centered matrix.
    """
    # The shared validator also computes correlations; very large finite input
    # values can overflow its intermediate calculations before our rejection.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        validate_rows(report)
    rows = report["rows"]
    by_rule = defaultdict(list)
    for i, row in enumerate(rows):
        by_rule[row["rule"]].append(i)
    x = np.asarray([[r["log_mean_marginal"], r["log_lift"],
                     r["log_lift"] * r["k_rel"]] for r in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Exposure features and interactions must be finite")
    centered = x.copy()
    for idx in by_rule.values():
        centered[idx] -= x[idx].mean(axis=0)
    rms = np.sqrt(np.mean(centered ** 2, axis=0))
    active = rms > 1e-12
    scaled = np.zeros_like(centered)
    scaled[:, active] = centered[:, active] / np.linalg.norm(centered[:, active], axis=0)
    _, singular, vh = np.linalg.svd(scaled, full_matrices=len(rows) < len(FEATURES))
    tol = max(singular[0] * 1e-10, 1e-12) if singular.size else 1e-12
    rank = int(np.count_nonzero(singular > tol))
    null = vh[rank:]
    findings = {}
    for i, name in enumerate(FEATURES):
        status = ("no_within_rule_variation" if not active[i] else
                  "collinear" if null.size and np.linalg.norm(null[:, i]) > 1e-8 else "identified")
        findings[name] = {"within_rule_rms": float(rms[i]), "status": status}
    within_lift = float(np.mean(centered[:, 1] ** 2))
    total_lift = float(np.var(x[:, 1]))
    ranges = [float(np.ptp(x[idx, 1])) for idx in by_rule.values()]
    return {"schema_version": 1, "rows": len(rows), "rules": len(by_rule),
            "features": findings, "rank": rank, "columns": len(FEATURES),
            "singular_values": singular.tolist(),
            "lift_variance": {"total": total_lift, "within_rule": within_lift,
                              "between_rule": max(0.0, total_lift - within_lift)},
            "median_rule_lift_range": float(np.median(ranges)),
            "gate_status": "not_assessed",
            "note": ("Structural diagnostic with fixed rule intercepts, not the proposal's "
                     "gate or a random-effects GLMM. Numerical rank depends on tolerance; "
                     "design and Monte Carlo uncertainty are not modeled.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = inspect(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
