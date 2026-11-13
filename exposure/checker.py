"""Prototype pixel-based checker calibrated for the current clean renderer.

This is not validated for generated, textured, overlapping or perturbed images.
"""

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from exposure.measure import load_rules, unique_object
from exposure.render import COLORS, IMAGE_SIZE, SHAPES, render_scene
from exposure.rules import Attribute, Predicate, Relation, SceneObject, satisfies, validate_rule


class DetectionError(ValueError):
    """The image is outside the checker's supported or confident range."""


class _PaletteError(DetectionError):
    """A pixel is incompatible with the supported renderer palette."""


def _detect_palette(image: Image.Image, threshold: float = 100, *, palette_tolerance: float = 40,
                    narrow_triangles: bool = False, guard_spatial: bool = True,
                    uncertain_out: list[bool] | None = None,
                    component_counts: tuple[int, ...] = (3,)) -> tuple[SceneObject, ...]:
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
    if np.any(blend_distance > palette_tolerance):
        raise _PaletteError("Unrecognized foreground colors")
    mask = distance <= threshold
    labels, count = ndimage.label(mask)
    if count not in component_counts:
        expected = "three" if component_counts == (3,) else "two, three or four"
        raise DetectionError(f"Expected {expected} separated objects; found {count} components")
    result = []
    uncertain = []
    for number in range(1, count + 1):
        ys, xs = np.nonzero(labels == number)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        width, height = x1 - x0 + 1, y1 - y0 + 1
        narrow = narrow_triangles and width == 11 and height == 11
        if not narrow and (width != 12 or height not in (11, 12)):
            raise DetectionError("Unsupported object size or occlusion")
        ratio = len(xs) / (width * height)
        if narrow and not 0.49 <= ratio <= 0.60:
            raise DetectionError("Ambiguous narrow object shape")
        if narrow or (height == 11 and 0.49 <= ratio <= 0.56):
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
        uncertain.append(narrow)
    if narrow_triangles and guard_spatial:
        _check_uncertain_centers(result, uncertain)
    if uncertain_out is not None:
        uncertain_out.extend(uncertain)
    return tuple(result)


def _check_uncertain_centers(objects: list[SceneObject], uncertain: list[bool]) -> None:
    """Conservative trial: reject potentially ambiguous x or y orderings.

    A one-pixel radius per narrow component is an unvalidated assumption, not
    a certified bound on generated images. It cannot establish shape validity.
    """
    for i, a in enumerate(objects):
        for j in range(i + 1, len(objects)):
            radius = int(uncertain[i]) + int(uncertain[j])
            if radius and (abs(a.x - objects[j].x) <= radius or abs(a.y - objects[j].y) <= radius):
                raise DetectionError("Uncertain spatial ordering")


def detect_narrow_experimental(image: Image.Image) -> tuple[SceneObject, ...]:
    """Trial narrow triangles without fallback; never validated for research use."""
    return _detect_palette(image, narrow_triangles=True)


def evaluate_narrow_attributes_experimental(image: Image.Image, predicates: tuple[Predicate, ...]) -> bool:
    """Evaluate attribute-only rules without requiring certain spatial ordering.

    This opt-in experiment does not validate object shapes or colors. Relation
    predicates are refused because uncertain centers can change their truth.
    """
    if not predicates or any(not isinstance(predicate, Attribute) for predicate in predicates):
        raise ValueError("Experimental attribute evaluation requires attribute-only predicates")
    objects = _detect_palette(image, narrow_triangles=True, guard_spatial=False)
    return satisfies(objects, predicates)


def _evaluate_uncertain(objects: tuple[SceneObject, ...], uncertain: list[bool],
                        predicates: tuple[Predicate, ...]) -> bool | None:
    """Conservative three-valued existential evaluation under assumed radii.

    None means the available center intervals do not settle the rule. The
    result is conservative: interacting uncertain relations may be infeasible,
    but will still abstain rather than inventing a definite answer.
    """
    validate_rule(predicates)
    if len(objects) != len(uncertain) or any(type(value) is not bool for value in uncertain):
        raise ValueError("Each object must have one uncertainty flag")
    names = sorted({name for p in predicates for name in
                    ((p.variable,) if isinstance(p, Attribute) else (p.subject, p.object))})
    possible = False
    for indices in permutations(range(len(objects)), len(names)):
        binding = dict(zip(names, indices, strict=True))
        unresolved = False
        for predicate in predicates:
            if isinstance(predicate, Attribute):
                if getattr(objects[binding[predicate.variable]], predicate.field) != predicate.value:
                    break
                continue
            i, j = binding[predicate.subject], binding[predicate.object]
            axis = "x" if predicate.kind == "left_of" else "y"
            a, b = getattr(objects[i], axis), getattr(objects[j], axis)
            radius = int(uncertain[i]) + int(uncertain[j])
            if a - b >= radius:
                break
            if a - b >= -radius:
                unresolved = True
        else:
            if not unresolved:
                return True
            possible = True
    return None if possible else False


def evaluate_narrow_rule_experimental(image: Image.Image, predicates: tuple[Predicate, ...]) -> bool | None:
    """Evaluate a rule with assumed ±1 pixel radii for narrow objects.

    This opt-in method returns None on uncertain relation truth and may raise
    DetectionError. Its center bounds and object labels are not validated.
    """
    flags: list[bool] = []
    objects = _detect_palette(image, narrow_triangles=True, guard_spatial=False, uncertain_out=flags)
    return _evaluate_uncertain(objects, flags, predicates)


def evaluate_variable_count_experimental(image: Image.Image, predicates: tuple[Predicate, ...]) -> bool | None:
    """Trial visible-witness evaluation on two, three or four components.

    With an unexpected component count, a definite positive witness can be
    returned, but missing or hidden objects mean a negative must abstain.
    Object detection and assumed center intervals remain unvalidated.
    """
    flags: list[bool] = []
    objects = _detect_palette(image, narrow_triangles=True, guard_spatial=False,
                              uncertain_out=flags, component_counts=(2, 3, 4))
    result = _evaluate_uncertain(objects, flags, predicates)
    return None if len(objects) != 3 and result is False else result


def _detect_sharpened(image: Image.Image, *, palette_tolerance: float = 40) -> tuple[SceneObject, ...]:
    """Accept mild sharpening only when two settings yield identical objects."""
    detections = [tuple(sorted(_detect_palette(image.filter(ImageFilter.UnsharpMask(
        radius=radius, percent=100, threshold=0)), palette_tolerance=palette_tolerance), key=lambda obj: (obj.x, obj.y)))
        for radius in (0.5, 0.6)]
    if detections[0] != detections[1]:
        raise DetectionError("Sharpening results disagree")
    return detections[0]


def detect(image: Image.Image, *, palette_tolerance: float = 40) -> tuple[SceneObject, ...]:
    """Infer objects from pixels with conservative fallback checks.

    The correction addresses a uniform grayscale reduction of the white
    background. It is deliberately narrow and does not imply general robustness.
    """
    if isinstance(palette_tolerance, bool) or not isinstance(palette_tolerance, (int, float)) or not np.isfinite(palette_tolerance) or not 0 <= palette_tolerance <= 45:
        raise ValueError("palette_tolerance must be finite and between 0 and 45")
    try:
        return _detect_palette(image, palette_tolerance=palette_tolerance)
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
                return _detect_palette(Image.fromarray(corrected.astype(np.uint8)), palette_tolerance=palette_tolerance)
            except DetectionError:
                pass
        # A second attempt is accepted only if two threshold
        # settings both classify all three components identically, including
        # exact recovered centers. Ambiguous/unsupported images still reject.
        try:
            detections = [tuple(sorted(_detect_palette(image, threshold, palette_tolerance=palette_tolerance), key=lambda obj: (obj.x, obj.y)))
                          for threshold in (115, 130)]
        except DetectionError:
            detections = []
        if detections and detections[0] == detections[1]:
            return detections[0]
        # Do not sharpen an image rejected by the original palette check.
        if not isinstance(original, _PaletteError):
            try:
                return _detect_sharpened(image, palette_tolerance=palette_tolerance)
            except DetectionError:
                pass
        raise original from None


def audit_rules() -> dict:
    """Fixed audit rules; include positive and negative examples when available."""
    rules = {f"color_{color}": (Attribute("a", "color", color),) for color in COLORS}
    rules.update({f"shape_{shape}": (Attribute("a", "shape", shape),) for shape in SHAPES})
    rules["red_left_of_square"] = (Attribute("a", "color", "red"),
                                    Attribute("b", "shape", "square"), Relation("a", "left_of", "b"))
    rules["blue_above_circle"] = (Attribute("a", "color", "blue"),
                                   Attribute("b", "shape", "circle"), Relation("a", "above", "b"))
    return rules


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _detection_metrics(cell: dict) -> dict:
    """Return conditional and full-distribution rates for one rule.

    Conditional rates use accepted images only. The unconditional rates count
    correct accepted classifications over all examples of the corresponding
    true class; abstentions are not counted as correct. These are coverage-
    adjusted correct-classification rates, not error rates with abstentions
    assigned a predicted class.
    """
    positives = cell["positive"] + cell["detector_failure_positive"]
    negatives = cell["negative"] + cell["detector_failure_negative"]
    return {
        "conditional_sensitivity": _ratio(cell["positive"] - cell["false_rejects"],
                                           cell["positive"]),
        "conditional_specificity": _ratio(cell["negative"] - cell["false_accepts"],
                                          cell["negative"]),
        "unconditional_sensitivity": _ratio(cell["positive"] - cell["false_rejects"],
                                            positives),
        "unconditional_specificity": _ratio(cell["negative"] - cell["false_accepts"],
                                            negatives),
        "positive_coverage": _ratio(cell["positive"], positives),
        "negative_coverage": _ratio(cell["negative"], negatives),
    }


def audit(samples: int, seed: int = 0, *, rules: dict | None = None) -> dict:
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
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
    for cell in results.values():
        cell.update(_detection_metrics(cell))
    accepted = samples - failed
    return {"seed_start": seed, "scenes": samples, "detector_failures": failed,
            "evaluated_scenes": accepted,
            "accepted_scenes": accepted,
            "rejection_rate": failed / samples,
            "coverage": accepted / samples,
            "rules": results,
            "note": ("Clean renderer only. Conditional rates use accepted images only; "
                     "unconditional rates count detector failures as missed positives. "
                     "No Stage 0 gate or human audit.")}


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
