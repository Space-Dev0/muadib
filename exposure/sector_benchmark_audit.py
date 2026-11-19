"""Independently audit a bounded-sector scene manifest, not model/checker quality."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from exposure.render import COLORS, SHAPES, _draw
from exposure.rules import SceneObject
from exposure.sector_benchmark import PARAMETERS, SCHEMA, rule_inventory


def _matches(objects: tuple[SceneObject, ...], rule: dict) -> bool:
    """Reference evaluator uses angle, independently of the generator's tangent test."""
    for i, subject in enumerate(objects):
        for j, target in enumerate(objects):
            if i == j or any(getattr(subject, k) != v for k, v in rule["subject"].items()):
                continue
            if any(getattr(target, k) != v for k, v in rule["target"].items()):
                continue
            dx, dy = subject.x - target.x, subject.y - target.y
            forward, lateral = (-dx, dy) if rule["direction"] == "left" else (-dy, dx)
            distance = math.hypot(dx, dy)
            if (forward > 0 and 14 <= distance <= 60
                    and abs(math.degrees(math.atan2(lateral, forward))) <= 15):
                return True
    return False


def audit(manifest: dict) -> dict:
    if (manifest.get("schema") != SCHEMA or manifest.get("parameters") != PARAMETERS
            or manifest.get("rules") != rule_inventory()):
        raise ValueError("Unexpected version, geometry, or rule inventory")
    count = manifest.get("scenes_per_rule")
    seeds = [manifest.get("train_seed"), manifest.get("heldout_seed")]
    budget = manifest.get("max_attempts")
    if (type(count) is not int or count < 1 or type(budget) is not int or budget < 1
            or any(type(seed) is not int or seed < 0 for seed in seeds) or seeds[0] == seeds[1]):
        raise ValueError("Invalid scene count, seeds, or attempt budget")
    rules = {rule["id"]: rule for rule in manifest["rules"]}
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Missing scene records")
    keys, hashes, attempts = set(), set(), []
    direction_counts: Counter[str] = Counter()
    for record in records:
        split, rule_id, index = record["split"], record["rule_id"], record["index"]
        key = (split, rule_id, index)
        if (split not in ("train", "heldout") or rule_id not in rules
                or type(index) is not int or not 0 <= index < count or key in keys):
            raise ValueError("Unexpected or duplicate scene slot")
        keys.add(key)
        attempt = record["attempts"]
        if type(attempt) is not int or not 1 <= attempt <= budget:
            raise ValueError("Invalid attempts count")
        attempts.append(attempt)
        raw = record["objects"]
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("Expected three annotated objects")
        objects = tuple(SceneObject(**item) for item in raw)
        grid_cells = set()
        for obj in objects:
            if (obj.color not in COLORS or obj.shape not in SHAPES
                    or type(obj.x) is not int or type(obj.y) is not int):
                raise ValueError("Invalid object attributes or coordinates")
            x_cells = [x for x in (12, 32, 52) if abs(obj.x - x) <= 3]
            y_cells = [y for y in (12, 32, 52) if abs(obj.y - y) <= 3]
            if len(x_cells) != 1 or len(y_cells) != 1:
                raise ValueError("Object lies outside the renderer grid")
            grid_cells.add((x_cells[0], y_cells[0]))
        if len(grid_cells) != 3 or not _matches(objects, rules[rule_id]):
            raise ValueError("Scene violates grid or requested rule")
        pixel_hash = hashlib.sha256(_draw(objects).image.tobytes()).hexdigest()
        if pixel_hash != record["pixel_sha256"] or pixel_hash in hashes:
            raise ValueError("Pixel hash mismatch or duplicated scene")
        hashes.add(pixel_hash)
        direction_counts[rules[rule_id]["direction"]] += 1
    expected = {(split, rule_id, i) for split in ("train", "heldout")
                for rule_id in rules for i in range(count)}
    if keys != expected:
        raise ValueError("Missing scene slots")
    return {"schema": SCHEMA, "rules": len(rules), "scenes": len(records),
            "scenes_per_split": len(records) // 2, "direction_counts": dict(direction_counts),
            "exact_pixel_duplicates": 0, "attempts_total": sum(attempts),
            "attempts_mean": statistics.mean(attempts), "attempts_median": statistics.median(attempts),
            "attempts_max": max(attempts),
            "note": "Positive-scene integrity audit only; not a checker, exposure-design, or Stage 0 pass."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(json.loads(args.manifest.read_text(encoding="utf-8")))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
