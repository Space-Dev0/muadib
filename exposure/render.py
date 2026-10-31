"""Generate simple, annotated synthetic scenes for instrument development.

Conditioned scenes are sampled by bounded rejection from the base distribution.
This prototype is not yet a validated exposure design for the study.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from exposure.rules import Attribute, Predicate, Relation, SceneObject, satisfies


COLORS = {
    "red": (220, 45, 45),
    "blue": (40, 100, 220),
    "green": (45, 160, 75),
    "yellow": (235, 190, 30),
}
SHAPES = ("circle", "square", "triangle")
IMAGE_SIZE = 64
SCALE = 4
RADIUS = 6


@dataclass(frozen=True)
class RenderedScene:
    image: Image.Image
    objects: tuple[SceneObject, ...]


def _sample_objects(rng: random.Random) -> tuple[SceneObject, ...]:
    """Sample objects from the prototype's unconditional scene distribution."""
    centers = [(x, y) for y in (12, 32, 52) for x in (12, 32, 52)]
    objects = []
    for base_x, base_y in rng.sample(centers, k=3):
        x, y = base_x + rng.randint(-3, 3), base_y + rng.randint(-3, 3)
        objects.append(SceneObject(color=rng.choice(tuple(COLORS)), shape=rng.choice(SHAPES),
                                   x=x, y=y))
    return tuple(objects)


def _draw(objects: tuple[SceneObject, ...]) -> RenderedScene:
    canvas = Image.new("RGB", (IMAGE_SIZE * SCALE, IMAGE_SIZE * SCALE), "white")
    draw = ImageDraw.Draw(canvas)
    for obj in objects:
        x, y, color, shape = obj.x, obj.y, obj.color, obj.shape
        left, top = (x - RADIUS) * SCALE, (y - RADIUS) * SCALE
        right, bottom = (x + RADIUS) * SCALE, (y + RADIUS) * SCALE
        if shape == "circle":
            draw.ellipse((left, top, right, bottom), fill=COLORS[color])
        elif shape == "square":
            draw.rectangle((left, top, right, bottom), fill=COLORS[color])
        else:
            draw.polygon(((x * SCALE, top), (left, bottom), (right, bottom)), fill=COLORS[color])
    image = canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return RenderedScene(image=image, objects=objects)


def render_scene(seed: int) -> RenderedScene:
    """Return a deterministic 64x64 RGB image and its object annotations."""
    return _draw(_sample_objects(random.Random(seed)))


def render_conditioned(seed: int, predicates: tuple[Predicate, ...],
                       max_attempts: int = 10_000) -> tuple[RenderedScene, int]:
    """Sample a matching scene, returning it and the number of attempts used.

    Failure within the budget does not establish that the rule is infeasible.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if not predicates:
        raise ValueError("A rule must contain at least one predicate")
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        objects = _sample_objects(rng)
        if satisfies(objects, predicates):
            return _draw(objects), attempt
    raise ValueError(f"No matching scene found in {max_attempts} attempts; "
                     "the rule may be rare or infeasible")


def parse_rule(data: object) -> tuple[Predicate, ...]:
    """Parse a JSON list of attribute and relation predicates."""
    if not isinstance(data, list) or not data:
        raise ValueError("Rule must be a nonempty JSON list")
    predicates = []
    for item in data:
        if not isinstance(item, dict) or not all(isinstance(v, str) and v for v in item.values()):
            raise ValueError("Each predicate must be an object with nonempty string values")
        if item.get("type") == "attribute" and set(item) == {"type", "variable", "field", "value"}:
            if item["value"] not in {"color": COLORS, "shape": SHAPES}.get(item["field"], ()):
                raise ValueError("Unsupported attribute field or value")
            predicates.append(Attribute(item["variable"], item["field"], item["value"]))
        elif item.get("type") == "relation" and set(item) == {"type", "subject", "kind", "object"}:
            predicates.append(Relation(item["subject"], item["kind"], item["object"]))
        else:
            raise ValueError("Invalid predicate type or fields")
    return tuple(predicates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one annotated synthetic scene")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducible output")
    parser.add_argument("--output", type=Path, default=Path("outputs/scene"),
                        help="output file prefix (writes .png and .json)")
    parser.add_argument("--rule", type=Path, help="JSON file containing predicate list")
    parser.add_argument("--max-attempts", type=int, default=10_000,
                        help="maximum samples for a conditioned rule")
    args = parser.parse_args()
    rule_data = None
    attempts = None
    if args.rule:
        try:
            rule_data = json.loads(args.rule.read_text(encoding="utf-8"))
            result, attempts = render_conditioned(args.seed, parse_rule(rule_data), args.max_attempts)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        result = render_scene(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(f"{args.output}.png")
    annotation_path = Path(f"{args.output}.json")
    result.image.save(image_path)
    annotation = {"seed": args.seed, "image_size": IMAGE_SIZE,
                  "objects": [asdict(obj) for obj in result.objects]}
    if rule_data is not None:
        annotation.update({"rule": rule_data, "attempts": attempts, "max_attempts": args.max_attempts})
    annotation_path.write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {image_path} and {annotation_path}")


if __name__ == "__main__":
    main()
