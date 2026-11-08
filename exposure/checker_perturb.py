"""Assess the prototype checker under fixed image perturbations.

This is an instrument diagnostic, not a model-output validation or a gate.
"""

import argparse
import json
from pathlib import Path

from PIL import ImageEnhance, ImageFilter

from exposure.checker import DetectionError, _detection_metrics, audit_rules, detect
from exposure.render import render_scene
from exposure.rules import satisfies


# Fixed a priori, applied separately (not cumulatively). Brightness affects the
# background as well as the objects; these are not a model of diffusion errors.
CASES = {
    "clean": ("clean", 0.0),
    "brightness_0.90": ("brightness", 0.9),
    "brightness_1.10": ("brightness", 1.1),
    "blur_0.50": ("blur", 0.5),
    "blur_1.00": ("blur", 1.0),
}


def perturb(image, case):
    kind, level = case
    if kind == "clean":
        return image.copy()
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(level)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=level))
    raise ValueError(f"Unsupported perturbation: {kind}")


def audit(samples: int, seed: int = 0) -> dict:
    """Audit paired scenes with known ground truth for each perturbation."""
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rules = audit_rules()
    results = {}
    for case_name, case in CASES.items():
        cells = {name: {"positive": 0, "negative": 0,
                        "detector_failure_positive": 0, "detector_failure_negative": 0,
                        "ground_truth_positive": 0, "ground_truth_negative": 0,
                        "false_accepts": 0, "false_rejects": 0} for name in rules}
        evaluated = rejected = 0
        for scene_seed in range(seed, seed + samples):
            scene = render_scene(scene_seed)
            try:
                detected = detect(perturb(scene.image, case))
            except DetectionError:
                detected = None
                rejected += 1
            else:
                evaluated += 1
            for name, predicates in rules.items():
                truth = satisfies(scene.objects, predicates)
                cell = cells[name]
                cell["ground_truth_positive" if truth else "ground_truth_negative"] += 1
                if detected is None:
                    cell["detector_failure_positive" if truth else "detector_failure_negative"] += 1
                    continue
                cell["positive" if truth else "negative"] += 1
                predicted = satisfies(detected, predicates)
                cell["false_accepts"] += int(predicted and not truth)
                cell["false_rejects"] += int(truth and not predicted)
        for cell in cells.values():
            cell.update(_detection_metrics(cell))
        results[case_name] = {"perturbation": {"kind": case[0], "level": case[1]},
                              "total": samples, "evaluated": evaluated,
                              "rejected": rejected,
                              "rejection_rate": rejected / samples,
                              "coverage": evaluated / samples, "rules": cells}
    return {"schema_version": 1, "seed_start": seed, "scenes_per_case": samples,
            "cases": results,
            "note": ("Paired clean synthetic renders only. Conditional rates use accepted "
                     "images only; unconditional rates count detector failures as missed "
                     "positives. No Stage 0 gate or human/model-output audit.")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit checker under image perturbations")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/checker-perturb.json"))
    args = parser.parse_args()
    try:
        result = audit(args.samples, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
