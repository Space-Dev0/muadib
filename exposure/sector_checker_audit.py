"""Development audit of a sector-rule adapter using clean synthetic scenes.

This is not a human-labeled or generated-image checker validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from exposure.checker import DetectionError, detect
from exposure.render import _draw, _sample_objects
from exposure.rules import SceneObject
from exposure.sector_benchmark_audit import _matches, audit


def run(manifest: dict, *, seed: int, max_attempts: int = 1000) -> dict:
    """Evaluate held-out positives and attribute-matched geometry negatives."""
    if type(seed) is not int or seed < 0 or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("seed must be nonnegative and max_attempts positive")
    audit(manifest)
    if seed in (manifest["train_seed"], manifest["heldout_seed"]):
        raise ValueError("negative sampling seed must differ from manifest seeds")
    rules = {rule["id"]: rule for rule in manifest["rules"]}
    seen = {record["pixel_sha256"] for record in manifest["records"]}
    rng = random.Random(seed)
    counts: dict[str, Counter] = {truth: Counter() for truth in ("positive", "negative")}
    records = []
    for record in manifest["records"]:
        if record["split"] != "heldout":
            continue
        rule = rules[record["rule_id"]]
        positive = tuple(SceneObject(**item) for item in record["objects"])
        for _ in range(max_attempts):
            positions = _sample_objects(rng)
            negative = tuple(SceneObject(obj.color, obj.shape, point.x, point.y)
                             for obj, point in zip(positive, positions, strict=True))
            if _matches(negative, rule):
                continue
            negative_hash = hashlib.sha256(_draw(negative).image.tobytes()).hexdigest()
            if negative_hash not in seen:
                seen.add(negative_hash)
                break
        else:
            raise RuntimeError(f"Unable to sample negative for {rule['id']}")
        for truth, objects in (("positive", positive), ("negative", negative)):
            image = _draw(objects).image
            try:
                decision = "yes" if _matches(detect(image), rule) else "no"
            except DetectionError:
                decision = "abstain"
            counts[truth][decision] += 1
            records.append({"rule_id": rule["id"], "index": record["index"],
                            "truth": truth, "decision": decision,
                            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    return {"schema": "sector-checker-audit-v1", "negative_seed": seed,
            "counts": {truth: {decision: counts[truth][decision]
                               for decision in ("yes", "no", "abstain")}
                       for truth in ("positive", "negative")},
            "records": records,
            "note": "Clean synthetic paired attribute-matched metadata; development only, not a validated checker."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        source = args.manifest.read_bytes()
        report = run(json.loads(source), seed=args.seed)
        report["manifest_sha256"] = hashlib.sha256(source).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
