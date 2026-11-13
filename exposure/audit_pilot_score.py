"""Score four restricted-pilot review sets without inventing consensus labels."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from exposure.audit_agreement import summarize
from exposure.human_audit import read_reviews, score
from exposure.measure import unique_object


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(pack: Path, reviews: dict[int, tuple[Path, Path]]) -> dict:
    """Score exact assigned IDs and compute overlap-only agreement.

    Each review pair is (whole-rule CSV, predicate CSV). A hash match verifies
    the current stored archive and private artifacts, not what reviewers saw.
    """
    if set(reviews) != {1, 2, 3, 4}:
        raise ValueError("Exactly four reviewer pairs, numbered 1–4, are required")
    private = pack / "private"
    provenance = _json(private / "provenance.json")
    required = {"all-rules.json", "all-predicates.json", "rule-predictions.json", "predicate-predictions.json"}
    for i in range(1, 5):
        required.update({f"reviewer-{i}-rules.json", f"reviewer-{i}-predicates.json",
                         f"reviewer-{i}-rule-predictions.json", f"reviewer-{i}-predicate-predictions.json"})
    hashes = provenance.get("frozen_private_sha256")
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise ValueError("Frozen private artifact list is missing or incomplete")
    for name, expected in hashes.items():
        if _digest(private / name) != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    archives = provenance.get("reviewer_archive_sha256")
    if not isinstance(archives, dict) or set(archives) != {f"reviewer-{i}.zip" for i in range(1, 5)}:
        raise ValueError("Frozen archive hash list is missing or incomplete")
    for name, expected in archives.items():
        if _digest(pack / name) != expected:
            raise ValueError(f"Reviewer archive hash mismatch: {name}")

    all_rules = _json(private / "all-rules.json")
    all_predicates = _json(private / "all-predicates.json")
    image_info = provenance.get("images")
    if not isinstance(image_info, dict):
        raise ValueError("Missing image hashes")
    for identifier, data in image_info.items():
        if not isinstance(identifier, str) or not re.fullmatch(r"pilot-\d{4,}", identifier):
            raise ValueError("Invalid image ID")
        image_path = (private / "images" / f"{identifier}.png").resolve(strict=True)
        if not image_path.is_relative_to((private / "images").resolve(strict=True)):
            raise ValueError("Image path escapes private image directory")
        if _digest(image_path) != data["sha256"]:
            raise ValueError(f"Image hash mismatch: {identifier}")
    shared = set(provenance.get("shared_ids", []))
    all_ids = {s["sample_id"] for s in all_rules["samples"]}
    if not shared or not shared < all_ids or set(image_info) != all_ids:
        raise ValueError("Invalid shared IDs or image inventory")
    shared_rule_manifest = {"samples": [s for s in all_rules["samples"] if s["sample_id"] in shared]}
    shared_predicate_manifest = {"samples": [s for s in all_predicates["samples"] if s["source_sample_id"] in shared]}
    shared_predicate_ids = {s["sample_id"] for s in shared_predicate_manifest["samples"]}

    scores = {}
    rule_shared = {}
    predicate_shared = {}
    assignments = Counter()
    review_hashes = {}
    for i in range(1, 5):
        name = f"reviewer-{i}"
        rule_path, predicate_path = reviews[i]
        rule_reviews, predicate_reviews = read_reviews(rule_path), read_reviews(predicate_path)
        rule_manifest = _json(private / f"{name}-rules.json")
        predicate_manifest = _json(private / f"{name}-predicates.json")
        rule_predictions = _json(private / f"{name}-rule-predictions.json")
        predicate_predictions = _json(private / f"{name}-predicate-predictions.json")
        scores[name] = {"rule": score(rule_manifest, rule_reviews, rule_predictions),
                        "predicate": score(predicate_manifest, predicate_reviews, predicate_predictions)}
        assigned = set(rule_reviews)
        if not shared <= assigned or not shared_predicate_ids <= set(predicate_reviews):
            raise ValueError(f"{name} is missing shared questions")
        assignments.update(assigned)
        rule_shared[name] = {key: rule_reviews[key] for key in shared}
        predicate_shared[name] = {key: predicate_reviews[key] for key in shared_predicate_ids}
        review_hashes[name] = {"rule": _digest(rule_path), "predicate": _digest(predicate_path)}
    if set(assignments) != all_ids or any(count != (4 if identifier in shared else 1)
                                           for identifier, count in assignments.items()):
        raise ValueError("Assignments do not cover each image as specified")
    return {"schema_version": 1, "unique_images_reviewed": len(assignments),
            "shared_images": len(shared), "rule_judgments": sum(assignments.values()),
            "duplicate_pixel_groups": len(provenance.get("duplicate_image_groups", [])),
            "review_sha256": review_hashes, "scores_by_reviewer": scores,
            "shared_rule_agreement": summarize(shared_rule_manifest, rule_shared),
            "shared_predicate_agreement": summarize(shared_predicate_manifest, predicate_shared),
            "note": ("No consensus or population accuracy is inferred. Shared and duplicate images are "
                     "not independent observations; per-reviewer scores overlap. The two-stratum pilot "
                     "is not the full Stage 0 audit. Hashes do not establish what reviewers actually saw.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for i in range(1, 5):
        parser.add_argument(f"--reviewer-{i}-rule", type=Path, required=True)
        parser.add_argument(f"--reviewer-{i}-predicate", type=Path, required=True)
    args = parser.parse_args()
    try:
        reviews = {i: (getattr(args, f"reviewer_{i}_rule"), getattr(args, f"reviewer_{i}_predicate"))
                   for i in range(1, 5)}
        result = evaluate(args.pack, reviews)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
