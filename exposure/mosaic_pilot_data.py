"""Generate a small, annotated MOSAIC-inspired 2x2 data feasibility pilot.

This creates training images, not trained models, and is not a replication.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from exposure.render import _draw
from exposure.rules import SceneObject

CELLS = (("low", "balanced"), ("low", "imbalanced"),
         ("high", "balanced"), ("high", "imbalanced"))


def generate(output: Path, *, seed: int, train_per_cell: int, eval_per_complexity: int) -> dict:
    """Create exact class quotas and an independent, complexity-balanced evaluation set.

    Low scenes have one target and one distractor; high scenes have the same
    target distribution and two distractors. The target is a red/blue circle;
    distractors are green/yellow squares or triangles. Training examples are
    freshly rendered, and the shared evaluation set is independent of every
    training cell. This restricted toy vocabulary is not MOSAIC's benchmark.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(train_per_cell) is not int or train_per_cell < 10 or train_per_cell % 10:
        raise ValueError("train_per_cell must be a positive multiple of 10, at least 10")
    if type(eval_per_complexity) is not int or eval_per_complexity < 2 or eval_per_complexity % 2:
        raise ValueError("eval_per_complexity must be a positive even integer")
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    seen: set[str] = set()
    records = []

    def write(split: str, complexity: str, imbalance: str | None, labels: list[str]) -> None:
        rng.shuffle(labels)
        for label in labels:
            # Reject exact duplicates across all cells and splits. Bounded so
            # impractical requests fail explicitly instead of looping forever.
            for _ in range(1000):
                positions = rng.sample([(x, y) for y in (12, 32, 52) for x in (12, 32, 52)],
                                       2 if complexity == "low" else 3)
                points = [(x + rng.randint(-3, 3), y + rng.randint(-3, 3)) for x, y in positions]
                objects = [SceneObject(label, "circle", *points[0])]
                objects.extend(SceneObject(rng.choice(("green", "yellow")),
                                           rng.choice(("square", "triangle")), *xy) for xy in points[1:])
                scene = _draw(tuple(objects))
                digest = hashlib.sha256(scene.image.tobytes()).hexdigest()
                if digest not in seen:
                    break
            else:
                raise RuntimeError("Unable to draw a unique scene within 1000 attempts")
            seen.add(digest)
            name = f"{len(records):06d}.png"
            scene.image.save(output / name)
            records.append({"image": name, "split": split, "complexity": complexity,
                            "imbalance": imbalance, "target_color": label,
                            "objects": [vars(obj) for obj in objects], "pixel_sha256": digest})

    for complexity, imbalance in CELLS:
        red = train_per_cell // 2 if imbalance == "balanced" else 9 * train_per_cell // 10
        write("train", complexity, imbalance, ["red"] * red + ["blue"] * (train_per_cell - red))
    for complexity in ("low", "high"):
        write("evaluation", complexity, None, ["red"] * (eval_per_complexity // 2)
              + ["blue"] * (eval_per_complexity // 2))
    manifest = {"schema_version": 1, "seed": seed, "train_per_cell": train_per_cell,
                "eval_per_complexity": eval_per_complexity, "records": records,
                "note": "Synthetic feasibility dataset only; no model, checker, or MOSAIC replication."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=81203)
    parser.add_argument("--train-per-cell", type=int, default=20)
    parser.add_argument("--eval-per-complexity", type=int, default=10)
    args = parser.parse_args()
    try:
        result = generate(args.output, seed=args.seed, train_per_cell=args.train_per_cell,
                          eval_per_complexity=args.eval_per_complexity)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(result['records'])} annotated feasibility scenes to {args.output}")


if __name__ == "__main__":
    main()
