"""Score the frozen 40-image checker validation without imputing disagreements."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from exposure.audit_pilot_score import _digest, _json, evaluate as evaluate_pilot
from exposure.human_audit import read_reviews, score

# These are the candidate predictions published before the fresh human review.
CANDIDATE_SHA256 = {
    "rule": "f895ea3443625dc74566715e132f4812eedd70b07d47f348c68eea950152e47f",
    "predicate": "d1f977854561d8cb11b14f4ebdcbc1422f6c5068b85485e9102c18054ed4b5f3",
}


def evaluate(pack: Path, reviews: dict[int, tuple[Path, Path]]) -> dict:
    """Validate the frozen pack and score each uniquely reviewed unanimous item.

    Single-reviewed items are retained, but are not presented as multi-reviewer
    consensus. Disputed items are listed and excluded rather than resolved.
    """
    validation = evaluate_pilot(pack, reviews)
    private = pack / "private"
    results = {}
    for index, (kind, plural) in enumerate((("rule", "rules"), ("predicate", "predicates"))):
        candidate_path = private / f"variable-count-{plural}-predictions.json"
        if _digest(candidate_path) != CANDIDATE_SHA256[kind]:
            raise ValueError(f"Frozen candidate hash mismatch: {candidate_path.name}")
        samples = _json(private / f"all-{plural}.json")["samples"]
        expected = {row["sample_id"] for row in samples}
        candidate = _json(candidate_path)
        standard = _json(private / f"{kind}-predictions.json")
        for name, predictions in (("standard", standard), ("candidate", candidate)):
            if (not isinstance(predictions, dict) or set(predictions) != expected or
                    any(value is not None and type(value) is not bool for value in predictions.values())):
                raise ValueError(f"Invalid {name} {kind} predictions")
        votes = defaultdict(list)
        for reviewer in range(1, 5):
            for identifier, answer in read_reviews(reviews[reviewer][index]).items():
                votes[identifier].append(answer)
        if set(votes) != expected:
            raise ValueError(f"Incomplete {kind} review coverage")
        disputed = sorted(identifier for identifier, values in votes.items() if len(set(values)) != 1)
        labels = {identifier: values[0] for identifier, values in votes.items() if len(set(values)) == 1}
        included = [row for row in samples if row["sample_id"] in labels]
        metrics = {}
        for name, predictions in (("standard", standard), ("candidate", candidate)):
            metrics[name] = (score({"samples": included}, labels,
                                   {key: predictions[key] for key in labels}) if included else None)
        results[kind] = {"unique_questions": len(expected), "unanimous_or_single_review": len(labels),
                         "single_reviewer_questions": sum(len(votes[k]) == 1 for k in labels),
                         "disputed_ids_excluded": disputed, "scores": metrics}
    return {"schema_version": 1, "pack_validation": validation, "results": results,
            "candidate_sha256": CANDIDATE_SHA256.copy(),
            "note": ("Pilot only: single-review labels are not consensus; disputed labels are excluded. "
                     "Seed blocks and multiple questions per image are dependent. Abstentions are not "
                     "errors; accepted error rates are conditional on selection. No gate or population "
                     "accuracy is established, and artifact hashes do not prove reviewer independence.")}


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
