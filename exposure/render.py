"""Generate simple, annotated synthetic scenes for instrument development.

This initial renderer samples scenes without conditioning on a rule. It is not
yet the template-conditioned sampling distribution required by the study.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from exposure.rules import SceneObject


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


def render_scene(seed: int) -> RenderedScene:
    """Return a deterministic 64x64 RGB image and its object annotations.

    Objects occupy different cells in a 3x3 grid. Their bounding boxes have at
    least a two-pixel gap even under the maximum positional jitter.
    """
    rng = random.Random(seed)
    canvas = Image.new("RGB", (IMAGE_SIZE * SCALE, IMAGE_SIZE * SCALE), "white")
    draw = ImageDraw.Draw(canvas)
    centers = [(x, y) for y in (12, 32, 52) for x in (12, 32, 52)]
    objects = []
    for base_x, base_y in rng.sample(centers, k=3):
        x, y = base_x + rng.randint(-3, 3), base_y + rng.randint(-3, 3)
        color, shape = rng.choice(tuple(COLORS)), rng.choice(SHAPES)
        obj = SceneObject(color=color, shape=shape, x=x, y=y)
        objects.append(obj)
        left, top = (x - RADIUS) * SCALE, (y - RADIUS) * SCALE
        right, bottom = (x + RADIUS) * SCALE, (y + RADIUS) * SCALE
        if shape == "circle":
            draw.ellipse((left, top, right, bottom), fill=COLORS[color])
        elif shape == "square":
            draw.rectangle((left, top, right, bottom), fill=COLORS[color])
        else:
            draw.polygon(((x * SCALE, top), (left, bottom), (right, bottom)), fill=COLORS[color])
    image = canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return RenderedScene(image=image, objects=tuple(objects))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one annotated synthetic scene")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducible output")
    parser.add_argument("--output", type=Path, default=Path("outputs/scene"),
                        help="output file prefix (writes .png and .json)")
    args = parser.parse_args()
    result = render_scene(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_path = Path(f"{args.output}.png")
    annotation_path = Path(f"{args.output}.json")
    result.image.save(image_path)
    annotation_path.write_text(json.dumps({"seed": args.seed, "image_size": IMAGE_SIZE,
        "objects": [asdict(obj) for obj in result.objects]}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {image_path} and {annotation_path}")


if __name__ == "__main__":
    main()
