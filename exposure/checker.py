"""Prototype pixel-based checker calibrated for the current clean renderer.

This is not validated for generated, textured, overlapping or perturbed images.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from exposure.measure import load_rules, unique_object
from exposure.render import COLORS, IMAGE_SIZE, SHAPES, render_scene
from exposure.rules import Attribute, Relation, SceneObject, satisfies


class DetectionError(ValueError):
    """The image is outside the checker's supported or confident range."""


def _detect_palette(image: Image.Image, threshold: float = 100) -> tuple[SceneObject, ...]:
    """Check pixels against the original clean-render palette."""
    if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise DetectionError("Expected a 64x64 RGB image")
    pixels = np.asarray(image, dtype=np.float64)
    palette = np.asarray(list(COLORS.values()), dtype=np.float64)
    distances = np.linalg.norm(pixels[:, :, None, :] - palette[None, None, :, :], axis=3)
    nearest = np.argmin(distances, axis=2)
    distance = np.min(distances, axis=2)
    # Antialiasing blends each palette color with the white background. Reject
    # pixels far from every such blend, allowing small rounding/ringing error.
    direction = palette - 255.0
    relative = pixels[:, :, None, :] - 255.0
    alpha = np.clip(np.sum(relative * direction, axis=3) / np.sum(direction * direction, axis=1), 0, 1)
    blend_distance = np.linalg.norm(relative - alpha[:, :, :, None] * direction, axis=3).min(axis=2)
    if np.any(blend_distance > 40):
        raise DetectionError("Unrecognized foreground colors")
    mask = distance <= threshold
    labels, count = ndimage.label(mask)
    if count != 3:
        raise DetectionError(f"Expected three separated objects; found {count} components")
    result = []
    for number in range(1, count + 1):
        ys, xs = np.nonzero(labels == number)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        width, height = x1 - x0 + 1, y1 - y0 + 1
        if width != 12 or height not in (11, 12):
            raise DetectionError("Unsupported object size or occlusion")
        ratio = len(xs) / (width * height)
        if height == 11 and 0.49 <= ratio <= 0.56:
            shape = "triangle"
        elif height == 12 and 0.76 <= ratio <= 0.84:
            shape = "circle"
        elif height == 12 and ratio >= 0.97:
            shape = "square"
        else:
            raise DetectionError("Ambiguous object shape")
        color_counts = np.bincount(nearest[ys, xs], minlength=len(COLORS))
        color_index = int(np.argmax(color_counts))
        if color_counts[color_index] / len(xs) < 0.95:
            raise DetectionError("Ambiguous object color")
        # The antialiased renderer's threshold mask occupies an even pixel span
        # for squares/circles, but an odd vertical span for triangles.
        x = (x0 + x1 + 1) / 2
        y = (y0 + y1 + (0 if shape == "triangle" else 1)) / 2
        result.append(SceneObject(tuple(COLORS)[color_index], shape, x, y))
    return tuple(result)


def detect(image: Image.Image) -> tuple[SceneObject, ...]:
    """Infer objects from pixels with conservative fallback checks.

    The correction addresses a uniform grayscale reduction of the white
    background. It is deliberately narrow and does not imply general robustness.
    """
    try:
        return _detect_palette(image)
    except DetectionError as original:
        if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise
        pixels = np.asarray(image, dtype=np.uint8)
        colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
        background = colors[int(np.argmax(counts))]
        # Only correct a dominant, neutral background in the calibrated range.
        if (counts.max() >= pixels.shape[0] * pixels.shape[1] * 0.5 and
                background[0] == background[1] == background[2] and
                225 <= int(background[0]) < 255):
            corrected = np.clip(np.rint(pixels.astype(np.float64) * 255.0 / int(background[0])), 0, 255)
            try:
                return _detect_palette(Image.fromarray(corrected.astype(np.uint8)))
            except DetectionError:
                pass
        # A second attempt is accepted only if two threshold
        # settings both classify all three components identically, including
        # exact recovered centers. Ambiguous/unsupported images still reject.
        try:
            detections = [tuple(sorted(_detect_palette(image, threshold), key=lambda obj: (obj.x, obj.y)))
                          for threshold in (115, 130)]
        except DetectionError:
            raise original from None
        if detections[0] != detections[1]:
            raise original
        return detections[0]


def audit_rules() -> dict:
    """Fixed audit rules; include positive and negative examples when available."""
    rules = {f"color_{color}": (Attribute("a", "color", color),) for color in COLORS}
    rules.update({f"shape_{shape}": (Attribute("a", "shape", shape),) for shape in SHAPES})
    rules["red_left_of_square"] = (Attribute("a", "color", "red"),
                                    Attribute("b", "shape", "square"), Relation("a", "left_of", "b"))
    rules["blue_above_circle"] = (Attribute("a", "color", "blue"),
                                   Attribute("b", "shape", "circle"), Relation("a", "above", "b"))
    return rules


def audit(samples: int, seed: int = 0, *, rules: dict | None = None) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    if rules is None:
        rules = audit_rules()
    if not rules:
        raise ValueError("Audit rules must not be empty")
    results = {name: {"positive": 0, "negative": 0, "false_accepts": 0, "false_rejects": 0,
                      "detector_failure_positive": 0, "detector_failure_negative": 0}
               for name in rules}
    failed = 0
    for scene_seed in range(seed, seed + samples):
        scene = render_scene(scene_seed)
        try:
            detected = detect(scene.image)
        except DetectionError:
            failed += 1
            for name, predicates in rules.items():
                truth = satisfies(scene.objects, predicates)
                results[name]["detector_failure_positive" if truth else "detector_failure_negative"] += 1
            continue
        for name, predicates in rules.items():
            truth = satisfies(scene.objects, predicates)
            predicted = satisfies(detected, predicates)
            cell = results[name]
            cell["positive" if truth else "negative"] += 1
            cell["false_accepts"] += bool(predicted and not truth)
            cell["false_rejects"] += bool(truth and not predicted)
    return {"seed_start": seed, "scenes": samples, "detector_failures": failed,
            "evaluated_scenes": samples - failed, "rules": results,
            "note": "Clean renderer only; failures excluded from rule denominators. No Stage 0 gate or human audit."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit image checker on clean synthetic renders")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rules", type=Path, help="optional JSON mapping rule names to predicate lists")
    parser.add_argument("--output", type=Path, default=Path("outputs/checker-audit.json"))
    args = parser.parse_args()
    try:
        rule_definitions = None
        rules = None
        if args.rules:
            rule_definitions = json.loads(args.rules.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
            rules = load_rules(rule_definitions)
        result = audit(args.samples, args.seed, rules=rules)
        if rule_definitions is not None:
            result["rule_definitions"] = rule_definitions
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
