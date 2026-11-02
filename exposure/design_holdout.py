"""Compare fixed exposure designs against a separate measurement report.

This exploratory diagnostic cannot establish a Stage 0 gate or statistical
confidence; it reports sensitivity to one additional estimated matrix.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from exposure.design_diagnostics import diagnose as design_diagnostics
from exposure.design_subspace import diagnose as subspace
from exposure.measure import unique_object


def compare(training: dict, holdout: dict, reference: dict, designs: dict) -> dict:
    """Evaluate fixed candidate weights against both measurement matrices."""
    if not isinstance(training, dict) or not isinstance(holdout, dict):
        raise ValueError("Both measurement reports must be objects")
    if training.get("rules") != holdout.get("rules") or training.get("rule_order") != holdout.get("rule_order"):
        raise ValueError("Measurement reports must use identical rules and ordering")
    if type(training.get("seed")) is not int or type(holdout.get("seed")) is not int:
        raise ValueError("Both measurement reports must record integer seeds")
    if training["seed"] == holdout["seed"]:
        raise ValueError("Training and holdout seeds must differ")
    fitted = subspace(training, reference)
    assessed = subspace(holdout, reference)
    if fitted["column_names"] != assessed["column_names"]:
        raise ValueError("Measurement condition order differs")
    first = fitted["singleton_events"]
    second = assessed["singleton_events"]
    if [item["event"] for item in first] != [item["event"] for item in second]:
        raise ValueError("Singleton events differ across reports")
    names = fitted["column_names"]
    reference_vector = np.array([fitted["reference_weights"][name] for name in names])
    training_matrix = np.array([event["column_probabilities"] for event in first])
    holdout_matrix = np.array([event["column_probabilities"] for event in second])
    training_diag = design_diagnostics(training, designs)
    holdout_diag = design_diagnostics(holdout, designs)
    drift = []
    for design in training_diag["designs"]:
        direction = np.array([design["weights"][name] for name in names]) - reference_vector
        train_residual = training_matrix @ direction
        holdout_residual = holdout_matrix @ direction
        drift.append({"name": design["name"],
                      "max_abs_training_deviation": float(np.max(np.abs(train_residual))),
                      "max_abs_holdout_deviation": float(np.max(np.abs(holdout_residual))),
                      "events": [{"event": first[i]["event"],
                                  "training_deviation": float(train_residual[i]),
                                  "holdout_deviation": float(holdout_residual[i])}
                                 for i in range(len(first))]})
    return {
        "schema_version": 1,
        "training_seed": training["seed"], "holdout_seed": holdout["seed"],
        "training_samples_per_condition": training.get("samples_per_condition"),
        "holdout_samples_per_condition": holdout.get("samples_per_condition"),
        "reference_weights": fitted["reference_weights"],
        "singleton_deviations": drift,
        "training": {k: training_diag[k] for k in ("row_count", "excluded_count", "correlations", "vif")},
        "holdout": {k: holdout_diag[k] for k in ("row_count", "excluded_count", "correlations", "vif")},
        "gate_status": "not_assessed",
        "note": ("Descriptive sensitivity to one independently seeded measurement, not a confidence "
                 "interval or Stage 0 gate. Designs are fixed from the training matrix; singleton "
                 "deviations compare each design with the reference within each matrix. Distinct "
                 "seeds alone do not prove independent scenes. Breadth remains omitted and rows "
                 "share designs and rules; excluded rows may differ across matrices."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--designs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output file")
    args = parser.parse_args()
    try:
        data = [json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
                for path in (args.training, args.holdout, args.reference, args.designs)]
        result = compare(*data)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(content)
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
