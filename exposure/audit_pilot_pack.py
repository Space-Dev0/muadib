"""Prepare blinded reviewer packs for the four-rule 800k pilot only.

This is a limited two-stratum pilot, not the full benchmark audit or a gate.
"""

import argparse
import csv
import hashlib
import html
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image
import PIL
import numpy
import scipy

from exposure import checker, human_audit, render, rules as rule_module
from exposure.checker import DetectionError, detect
from exposure.human_audit import predicate_manifest
from exposure.render import parse_rule
from exposure.rules import satisfies


QUESTIONS = {
    "red": "Is there a red object?",
    "square": "Is there a square?",
    "red_left_of_square": "Is a red object to the left of a different square object?",
    "blue_above_circle": "Is a blue object above a different circular object?",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv(ids: list[str]) -> str:
    result = io.StringIO(newline="")
    writer = csv.writer(result, lineterminator="\n")
    writer.writerow(["sample_id", "judgment"])
    writer.writerows((identifier, "") for identifier in ids)
    return result.getvalue()


def _predicate_question(sample: dict) -> str:
    predicate = sample["predicates"][0]
    if predicate["type"] == "attribute":
        return f"Is there an object whose {predicate['field']} is {predicate['value']}?"
    relation = "to the left of" if predicate["kind"] == "left_of" else "above"
    return f"Are there two distinct objects with one {relation} the other?"


def _page(rule_rows: list[dict], predicate_rows: list[dict]) -> str:
    def table(rows: list[dict], question) -> str:
        return "".join(
            f'<tr><td>{html.escape(row["sample_id"])}</td><td><img src="{html.escape(row["image_path"], quote=True)}" '
            f'width="192" height="192"></td><td>{html.escape(question(row))}</td></tr>' for row in rows)

    header = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
              "<title>Pilot image review</title><style>body{font:16px sans-serif;max-width:950px;margin:auto}"
              "td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}img{image-rendering:pixelated}</style>"
              "<h1>Pilot image review</h1><p>Work independently. For each question enter yes or no in its "
              "CSV file, matching the sample ID; preserve the headers and IDs. Save both CSVs under your "
              "reviewer name. Note ambiguous cases separately. Do not discuss decisions until everyone has "
              "finished. Questions involving two objects require distinct objects. For separate predicate "
              "questions, judge only the question shown, independently of any other question on the image.</p>"
              "<p>This is a restricted four-condition pilot, not the full benchmark audit.</p>")
    headings = "<table><tr><th>ID</th><th>Image</th><th>Question</th></tr>"
    return (header + "<h2>Whole-rule questions: rule-reviews.csv</h2>" + headings
            + table(rule_rows, lambda row: QUESTIONS[row["rule_name"]]) + "</table>"
            + "<h2>Predicate questions: predicate-reviews.csv</h2>" + headings
            + table(predicate_rows, _predicate_question) + "</table></html>")


def build(source: Path, output: Path, checkpoint: Path, *, seed: int = 81213,
          samples_per_rule: int = 75, shared_per_rule: int = 10, first_seed: int = 20000) -> dict:
    """Package exactly specified samples with four balanced assignments and private predictions."""
    for value in (seed, samples_per_rule, shared_per_rule, first_seed):
        if type(value) is not int or value < 0:
            raise ValueError("Seeds and counts must be nonnegative integers")
    if samples_per_rule < 1 or not 0 < shared_per_rule < samples_per_rule:
        raise ValueError("Shared count must be positive and smaller than each condition")
    source = source.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    if not source.is_dir() or not checkpoint.is_file():
        raise ValueError("Source must be a directory and checkpoint a file")
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError("Output and image source must be separate directories")

    # Freeze definitions independent of any local example file that may change.
    definitions = {
        "red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}],
        "square": [{"type": "attribute", "variable": "b", "field": "shape", "value": "square"}],
        "red_left_of_square": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"},
                               {"type": "attribute", "variable": "b", "field": "shape", "value": "square"},
                               {"type": "relation", "subject": "a", "kind": "left_of", "object": "b"}],
        "blue_above_circle": [{"type": "attribute", "variable": "a", "field": "color", "value": "blue"},
                              {"type": "attribute", "variable": "b", "field": "shape", "value": "circle"},
                              {"type": "relation", "subject": "a", "kind": "above", "object": "b"}],
    }
    items = []
    for rule, predicates in definitions.items():
        for generation_seed in range(first_seed, first_seed + samples_per_rule):
            path = (source / f"{rule}-{generation_seed}.png").resolve(strict=True)
            if not path.is_relative_to(source) or not path.is_file():
                raise ValueError("Image path escapes the source directory")
            source_data = path.read_bytes()
            with Image.open(io.BytesIO(source_data)) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB" or image.size != (64, 64):
                    raise ValueError(f"Invalid image: {path.name}")
                pixel_bytes = image.tobytes()
                pixel_hash = _sha(pixel_bytes)
                # Re-encode pixels without source PNG metadata that could reveal condition or provenance.
                clean = Image.frombytes("RGB", image.size, pixel_bytes)
                encoded = io.BytesIO()
                clean.save(encoded, format="PNG")
                data = encoded.getvalue()
            items.append({"rule": rule, "predicates": predicates, "generation_seed": generation_seed,
                          "bytes": data, "sha256": _sha(data), "pixel_sha256": pixel_hash,
                          "source_sha256": _sha(source_data), "source_name": path.name})

    rng = random.Random(seed)
    rng.shuffle(items)
    for index, item in enumerate(items, 1):
        item["id"] = f"pilot-{index:04d}"
    groups = {rule: [item for item in items if item["rule"] == rule] for rule in QUESTIONS}
    assignments = [[] for _ in range(4)]
    shared = []
    offset = 0
    for rule in QUESTIONS:
        group = groups[rule]
        shared.extend(item["id"] for item in group[:shared_per_rule])
        for index, item in enumerate(group[shared_per_rule:]):
            assignments[(offset + index) % 4].append(item["id"])
        offset = (offset + len(group) - shared_per_rule) % 4
    by_id = {item["id"]: item for item in items}
    samples = [{"sample_id": item["id"], "image_path": f'images/{item["id"]}.png',
                "rule_name": item["rule"], "predicates": item["predicates"],
                "k_attr": sum(p["type"] == "attribute" for p in item["predicates"]),
                "k_rel": sum(p["type"] == "relation" for p in item["predicates"])} for item in items]

    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir(mode=0o700)
    images = private / "images"
    images.mkdir()
    for item in items:
        (images / f'{item["id"]}.png').write_bytes(item["bytes"])
    manifest = {"samples": samples}
    predicates = predicate_manifest(manifest, private)
    by_image = defaultdict(list)
    for row in predicates["samples"]:
        by_image[row["source_sample_id"]].append(row)

    def predictions(rows: list[dict]) -> dict:
        result = {}
        for row in rows:
            try:
                with Image.open(private / row["image_path"]) as image:
                    result[row["sample_id"]] = satisfies(detect(image), parse_rule(row["predicates"]))
            except DetectionError:
                result[row["sample_id"]] = None
        return result

    rule_predictions = predictions(samples)
    predicate_predictions = predictions(predicates["samples"])
    private_data = {
        "schema_version": 1, "seed": seed, "first_seed": first_seed,
        "checkpoint_sha256": _sha(checkpoint.read_bytes()),
        "source_hashes": {p.__name__: _sha(Path(p.__file__).read_bytes()) for p in (checker, human_audit, render, rule_module)},
        "packager_sha256": _sha(Path(__file__).read_bytes()),
        "runtime": {"numpy": numpy.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__},
        "images": {item["id"]: {"sha256": item["sha256"], "pixel_sha256": item["pixel_sha256"],
                                    "source_sha256": item["source_sha256"], "source_name": item["source_name"],
                                    "generation_seed": item["generation_seed"]} for item in items},
        "shared_ids": shared,
        "note": ("Restricted pilot; checkpoint hash identifies supplied file, not verified image origin. "
                 "No labels, gate result, or verified reviewer independence."),
    }
    duplicate_hashes = defaultdict(list)
    for item in items:
        duplicate_hashes[item["pixel_sha256"]].append(item["id"])
    private_data["duplicate_image_groups"] = [ids for ids in duplicate_hashes.values() if len(ids) > 1]
    for filename, value in (("all-rules.json", manifest), ("all-predicates.json", predicates),
                            ("rule-predictions.json", rule_predictions), ("predicate-predictions.json", predicate_predictions)):
        (private / filename).write_text(json.dumps(value, indent=2) + "\n")

    for index, exclusive in enumerate(assignments, 1):
        selected = shared + exclusive
        rng.shuffle(selected)
        selected_set = set(selected)
        rule_rows = [row for identifier in selected for row in samples if row["sample_id"] == identifier]
        pred_rows = [row for identifier in selected for row in by_image[identifier]]
        private_manifest = {"samples": rule_rows}
        private_predicates = {"samples": pred_rows}
        (private / f"reviewer-{index}-rules.json").write_text(json.dumps(private_manifest, indent=2) + "\n")
        (private / f"reviewer-{index}-predicates.json").write_text(json.dumps(private_predicates, indent=2) + "\n")
        # Per-reviewer predictions can be scored without exposing them to the reviewer.
        for kind, rows, mapping in (("rule", rule_rows, rule_predictions), ("predicate", pred_rows, predicate_predictions)):
            (private / f"reviewer-{index}-{kind}-predictions.json").write_text(
                json.dumps({row["sample_id"]: mapping[row["sample_id"]] for row in rows}, indent=2) + "\n")
        with zipfile.ZipFile(output / f"reviewer-{index}.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("review.html", _page(rule_rows, pred_rows))
            archive.writestr("rule-reviews.csv", _csv([row["sample_id"] for row in rule_rows]))
            archive.writestr("predicate-reviews.csv", _csv([row["sample_id"] for row in pred_rows]))
            for identifier in sorted(selected_set):
                archive.writestr(f"images/{identifier}.png", by_id[identifier]["bytes"])
    private_data["frozen_private_sha256"] = {
        path.name: _sha(path.read_bytes()) for path in private.iterdir() if path.is_file()
    }
    private_data["reviewer_archive_sha256"] = {
        f"reviewer-{index}.zip": _sha((output / f"reviewer-{index}.zip").read_bytes())
        for index in range(1, 5)
    }
    (private / "provenance.json").write_text(json.dumps(private_data, indent=2) + "\n")
    return {"images": len(items), "shared_images": len(shared),
            "images_per_reviewer": [len(shared) + len(group) for group in assignments],
            "duplicate_image_groups": len(private_data["duplicate_image_groups"]),
            "note": private_data["note"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=81213)
    args = parser.parse_args()
    try:
        result = build(args.source, args.output, args.checkpoint, seed=args.seed)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
