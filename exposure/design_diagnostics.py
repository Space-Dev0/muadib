"""Exploratory pooled exposure-design collinearity diagnostics.

Rows are correlated rule-by-design observations. This is not a Stage 0 gate.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from exposure.measure import load_rules, unique_object
from exposure.mixture import summarize
from exposure.rules import Attribute, Relation

FEATURES = ("log_mean_marginal", "log_lift", "k_attr", "k_rel")


def statistics(rows: list[dict]) -> dict:
    """Return pointwise Pearson correlations and per-feature VIFs.

    VIF regressions include an intercept. Collinear and unidentifiable values
    are null rather than infinite; no threshold constitutes a research gate.
    """
    count = len(rows)
    matrix = np.asarray([[row[feature] for feature in FEATURES] for row in rows], dtype=float)
    if count == 0:
        matrix = np.empty((0, len(FEATURES)))
    correlations = {}
    for i, first in enumerate(FEATURES):
        correlations[first] = {}
        for j, second in enumerate(FEATURES):
            value = None
            if count >= 2 and np.std(matrix[:, i]) > 1e-12 and np.std(matrix[:, j]) > 1e-12:
                value = float(np.clip(np.corrcoef(matrix[:, i], matrix[:, j])[0, 1], -1, 1))
            correlations[first][second] = value
    vifs = {}
    for i, feature in enumerate(FEATURES):
        if count <= len(FEATURES):
            vifs[feature] = {"value": None, "status": "insufficient_rows"}
            continue
        target = matrix[:, i]
        centered = target - target.mean()
        total = float(centered @ centered)
        if total <= count * 1e-24:
            vifs[feature] = {"value": None, "status": "constant"}
            continue
        predictors = np.delete(matrix, i, axis=1)
        deviations = predictors.std(axis=0)
        # A constant predictor contributes nothing beyond the intercept.
        predictors = predictors[:, deviations > 1e-12]
        if predictors.shape[1]:
            predictors = (predictors - predictors.mean(axis=0)) / predictors.std(axis=0)
        design = np.column_stack((np.ones(count), predictors))
        fitted = design @ np.linalg.lstsq(design, centered, rcond=None)[0]
        residual = centered - fitted
        fraction = float(residual @ residual) / total
        if fraction <= 1e-10:
            vifs[feature] = {"value": None, "status": "collinear_or_near_collinear"}
        else:
            vifs[feature] = {"value": max(1.0, 1.0 / fraction), "status": "ok"}
    return {"correlations": correlations, "vif": vifs}


def diagnose(report: dict, configuration: dict) -> dict:
    """Calculate statistics for a named collection of candidate designs."""
    if not isinstance(configuration, dict) or set(configuration) != {"designs"}:
        raise ValueError("Configuration must contain only a designs list")
    designs = configuration["designs"]
    if not isinstance(designs, list) or len(designs) < 3:
        raise ValueError("At least three candidate designs are required")
    rules = load_rules(report.get("rules") if isinstance(report, dict) else None)
    identifiers = set()
    rows, excluded, normalized_designs = [], [], []
    for design in designs:
        if not isinstance(design, dict) or set(design) != {"name", "weights"}:
            raise ValueError("Each design requires only name and weights")
        name = design["name"]
        if not isinstance(name, str) or not name.strip() or name in identifiers:
            raise ValueError("Design names must be unique nonempty strings")
        identifiers.add(name)
        summary = summarize(report, design["weights"])
        normalized_designs.append({"name": name, "weights": summary["weights"]})
        for rule, predicates in rules.items():
            result = summary["results"][rule]
            marginal = math.fsum(result["predicate_marginals"]) / len(predicates)
            lift = result["log_lift"]
            if not math.isfinite(marginal) or marginal <= 0:
                excluded.append({"design": name, "rule": rule, "reason": "zero_or_nonfinite_mean_marginal"})
            elif lift is None or not math.isfinite(lift):
                excluded.append({"design": name, "rule": rule, "reason": result["lift_status"]})
            else:
                rows.append({"design": name, "rule": rule, "log_mean_marginal": math.log(marginal),
                             "log_lift": lift,
                             "k_attr": sum(isinstance(predicate, Attribute) for predicate in predicates),
                             "k_rel": sum(isinstance(predicate, Relation) for predicate in predicates)})
    return {"schema_version": 1, "designs": normalized_designs, "source_seed": report.get("seed"),
            "features": list(FEATURES), "rows": rows, "row_count": len(rows),
            "excluded_count": len(excluded), "excluded": excluded, **statistics(rows),
            "gate_status": "not_assessed", "note": (
                "Exploratory pooled rule-by-design rows share rules and designs; exposures are "
                "Monte Carlo estimates. Breadth is omitted, uncertainty is not propagated, "
                "and this does not assess the proposal's decorrelation gate or causality.")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore candidate exposure-design covariates")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--designs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/design-diagnostics.json"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        designs = json.loads(args.designs.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = diagnose(report, designs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
