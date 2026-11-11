"""Prepare blinded rule reviews and score human judgments against predictions.

This tool never creates human labels. Rejected predictions are reported
separately and are excluded from conditional false-accept/reject denominators.
"""

import argparse
import csv
import json
import os
import random
import tempfile
from pathlib import Path

from PIL import Image

from exposure.checker import DetectionError, detect
from exposure.measure import unique_object
from exposure.render import parse_rule
from exposure.rules import Attribute, Relation, satisfies


def _samples(manifest: object) -> list[dict]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list) or not manifest["samples"]:
        raise ValueError("Manifest must contain a nonempty samples list")
    result, ids, rule_definitions = [], set(), {}
    for item in manifest["samples"]:
        required = {"sample_id", "image_path", "rule_name", "predicates", "k_attr", "k_rel"}
        if not isinstance(item, dict) or not required <= set(item):
            raise ValueError("Each sample must contain ID, image path, rule, predicates and complexity")
        sample_id, image_path, name = (item[key] for key in ("sample_id", "image_path", "rule_name"))
        if any(not isinstance(value, str) or not value.strip() for value in (sample_id, image_path, name)):
            raise ValueError("Sample ID, image path and rule name must be nonempty strings")
        if sample_id in ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        ids.add(sample_id)
        predicates = parse_rule(item["predicates"])
        k_attr, k_rel = item["k_attr"], item["k_rel"]
        if (type(k_attr) is not int or type(k_rel) is not int or
                k_attr != sum(isinstance(p, Attribute) for p in predicates) or
                k_rel != sum(isinstance(p, Relation) for p in predicates)):
            raise ValueError(f"Incorrect complexity counts for sample {sample_id}")
        encoded = json.dumps(item["predicates"], sort_keys=True, separators=(",", ":"))
        if name in rule_definitions and rule_definitions[name] != encoded:
            raise ValueError(f"Inconsistent predicates for rule {name}")
        rule_definitions[name] = encoded
        result.append({key: item[key] for key in ("sample_id", "image_path", "rule_name", "predicates", "k_attr", "k_rel")})
    return result


def _root(root: Path) -> Path:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Dataset root must be a directory")
    return root


def _image(root: Path, image_path: str) -> tuple[Path, str]:
    path = Path(image_path)
    if path.is_absolute():
        raise ValueError("Image paths must be relative to the dataset root")
    resolved = (root / path).resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Image path escapes dataset root") from exc
    if not resolved.is_file():
        raise ValueError("Image path must point to a file")
    return resolved, relative.as_posix()


def prepare(manifest: object, root: Path, *, per_stratum: int, seed: int) -> dict:
    """Return a reproducible, stratified manifest without predictions or labels."""
    if type(per_stratum) is not int or per_stratum < 1 or type(seed) is not int:
        raise ValueError("per_stratum must be positive and seed must be an integer")
    root = _root(root)
    strata: dict[tuple[int, int], list[dict]] = {}
    for sample in _samples(manifest):
        _, sample["image_path"] = _image(root, sample["image_path"])
        strata.setdefault((sample["k_attr"], sample["k_rel"]), []).append(sample)
    rng = random.Random(seed)
    selected = []
    for stratum, candidates in sorted(strata.items()):
        if len(candidates) < per_stratum:
            raise ValueError(f"Stratum {stratum} has fewer than {per_stratum} samples")
        candidates.sort(key=lambda item: item["sample_id"])
        selected.extend(rng.sample(candidates, per_stratum))
    return {"schema_version": 1, "seed": seed, "per_stratum": per_stratum, "samples": selected,
            "note": "Blinded review manifest; no human judgments are supplied."}


def predicate_manifest(manifest: object, root: Path) -> dict:
    """Expand reviewed rules into separate, independently rebound predicate questions.

    Source IDs link the questions to the original complexity-stratified sample;
    neither rule judgments nor checker predictions are copied or inferred.
    """
    root = _root(root)
    questions = []
    for sample in sorted(_samples(manifest), key=lambda item: item["sample_id"]):
        _, image_path = _image(root, sample["image_path"])
        for index, predicate in enumerate(parse_rule(sample["predicates"])):
            if isinstance(predicate, Attribute):
                name = f"attribute:{predicate.field}:{predicate.value}"
                atom = {"type": "attribute", "variable": "a", "field": predicate.field,
                        "value": predicate.value}
                k_attr, k_rel = 1, 0
            else:
                name = f"relation:{predicate.kind}"
                atom = {"type": "relation", "subject": "a", "kind": predicate.kind, "object": "b"}
                k_attr, k_rel = 0, 1
            questions.append({"sample_id": f"predicate-{len(questions) + 1:06d}",
                              "image_path": image_path, "rule_name": name, "predicates": [atom],
                              "k_attr": k_attr, "k_rel": k_rel,
                              "source_sample_id": sample["sample_id"], "predicate_index": index})
    return {"schema_version": 1, "samples": questions, "note": (
        "Each question independently rebinds its predicate. Collect separate human judgments; "
        "a whole-rule judgment cannot be used as a predicate judgment.")}


def predict(manifest: object, root: Path) -> dict[str, bool | None]:
    """Produce checker predictions without generating human judgments.

    Only a detector rejection becomes null. Missing, unreadable, or invalid
    image files abort the operation so input errors cannot look like rejection.
    """
    root = _root(root)
    predictions = {}
    for sample in _samples(manifest):
        path, _ = _image(root, sample["image_path"])
        with Image.open(path) as image:
            image.load()
            try:
                objects = detect(image)
            except DetectionError:
                predictions[sample["sample_id"]] = None
                continue
            predictions[sample["sample_id"]] = satisfies(objects, parse_rule(sample["predicates"]))
    return predictions


def read_reviews(path: Path) -> dict[str, bool]:
    """Read reviewer CSV columns sample_id and judgment, with yes/no judgments."""
    judgments = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["sample_id", "judgment"]:
            raise ValueError("Reviewer CSV must have sample_id,judgment header")
        for row in reader:
            identifier, answer = row.get("sample_id"), row.get("judgment")
            if not identifier or identifier in judgments or answer is None or answer.strip().lower() not in {"yes", "no"}:
                raise ValueError("Invalid or duplicate reviewer judgment")
            if None in row:
                raise ValueError("Reviewer row has extra fields")
            judgments[identifier] = answer.strip().lower() == "yes"
    return judgments


def _metrics() -> dict:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "rejected_human_yes": 0, "rejected_human_no": 0}


def _add(metrics: dict, human: bool, prediction: bool | None) -> None:
    if prediction is None:
        key = "rejected_human_yes" if human else "rejected_human_no"
    elif human:
        key = "tp" if prediction else "fn"
    else:
        key = "fp" if prediction else "tn"
    metrics[key] += 1


def _finish(metrics: dict) -> dict:
    positives, negatives = metrics["tp"] + metrics["fn"], metrics["tn"] + metrics["fp"]
    human_yes = positives + metrics["rejected_human_yes"]
    human_no = negatives + metrics["rejected_human_no"]
    total = human_yes + human_no
    return {**metrics, "accepted_human_yes": positives, "accepted_human_no": negatives,
            "false_reject_rate": metrics["fn"] / positives if positives else None,
            "false_accept_rate": metrics["fp"] / negatives if negatives else None,
            "coverage": (positives + negatives) / total if total else None,
            "coverage_human_yes": positives / human_yes if human_yes else None,
            "coverage_human_no": negatives / human_no if human_no else None,
            "false_reject_completion_bounds": ([metrics["fn"] / human_yes,
                (metrics["fn"] + metrics["rejected_human_yes"]) / human_yes] if human_yes else None),
            "false_accept_completion_bounds": ([metrics["fp"] / human_no,
                (metrics["fp"] + metrics["rejected_human_no"]) / human_no] if human_no else None)}


def score(manifest: object, judgments: dict[str, bool], predictions: object) -> dict:
    """Compare exact reviewed IDs with independent true/false/null predictions."""
    samples = _samples(manifest)
    expected = {sample["sample_id"] for sample in samples}
    if (not isinstance(judgments, dict) or set(judgments) != expected or
            any(type(value) is not bool for value in judgments.values())):
        raise ValueError("Reviewer judgments must contain exactly the manifest IDs")
    if (not isinstance(predictions, dict) or set(predictions) != expected or
            any(value is not None and type(value) is not bool for value in predictions.values())):
        raise ValueError("Predictions must contain exactly the manifest IDs with true, false or null values")
    overall, by_rule, by_stratum = _metrics(), {}, {}
    for sample in samples:
        identifier, rule = sample["sample_id"], sample["rule_name"]
        stratum = f'{sample["k_attr"]},{sample["k_rel"]}'
        human, prediction = judgments[identifier], predictions[identifier]
        for target in (overall, by_rule.setdefault(rule, _metrics()), by_stratum.setdefault(stratum, _metrics())):
            _add(target, human, prediction)
    return {"schema_version": 1, "reviewed": len(samples), "overall": _finish(overall),
            "by_rule": {name: _finish(value) for name, value in sorted(by_rule.items())},
            "by_stratum": {name: _finish(value) for name, value in sorted(by_stratum.items())},
            "note": ("Conditional rates exclude rejected predictions. Completion bounds assign each "
                     "rejection an arbitrary binary outcome and describe the reviewed sample only; "
                     "they are not confidence intervals or actual error rates. No automatic gate.")}


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def _image_inputs(manifest: object, root: Path) -> list[Path]:
    """Resolve image inputs with the same containment checks used for reads."""
    root = _root(root)
    return [_image(root, item["image_path"])[0] for item in _samples(manifest)]


def _protect_output(output: Path, input_paths: list[Path]) -> None:
    """Reject direct, symlink or existing hardlink aliases of input files."""
    for source in input_paths:
        if (output.resolve() == source.resolve() or
                (output.exists() and output.samefile(source))):
            raise ValueError("Output must differ from all input files")


def _write_json(output: Path, result: object) -> None:
    """Replace an output atomically, leaving an existing file intact on failure."""
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".human-audit-", delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and score a blinded human audit")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--dataset-root", type=Path, required=True)
    create.add_argument("--per-stratum", type=int, required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--output", type=Path, required=True)
    expand = commands.add_parser("predicates", help="prepare independent predicate questions")
    expand.add_argument("--manifest", type=Path, required=True)
    expand.add_argument("--dataset-root", type=Path, required=True)
    expand.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("predict")
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--dataset-root", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("score")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--reviews", type=Path, required=True)
    compare.add_argument("--predictions", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            manifest = _json(args.manifest)
            _protect_output(args.output, [args.manifest, *_image_inputs(manifest, args.dataset_root)])
            result = prepare(manifest, args.dataset_root, per_stratum=args.per_stratum, seed=args.seed)
        elif args.command == "predicates":
            manifest = _json(args.manifest)
            _protect_output(args.output, [args.manifest, *_image_inputs(manifest, args.dataset_root)])
            result = predicate_manifest(manifest, args.dataset_root)
        elif args.command == "predict":
            manifest = _json(args.manifest)
            _protect_output(args.output, [args.manifest, *_image_inputs(manifest, args.dataset_root)])
            result = predict(manifest, args.dataset_root)
        else:
            _protect_output(args.output, [args.manifest, args.reviews, args.predictions])
            result = score(_json(args.manifest), read_reviews(args.reviews), _json(args.predictions))
        _write_json(args.output, result)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
