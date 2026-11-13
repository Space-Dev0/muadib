"""Apply selected follow-up answers separately from original pilot reviews.

Follow-up cases were selected after initial review and are not independent
validation. This helper never changes the original human judgments.
"""

import argparse
import csv
import json
from pathlib import Path

from exposure.human_audit import _samples, read_reviews, score
from exposure.measure import unique_object


def combine(manifest: object, reviews: list[dict], predictions: dict, followup: list[dict]) -> dict:
    """Build a sensitivity score, refusing missing or ambiguous decisions."""
    samples = _samples(manifest)
    ids = {item["sample_id"] for item in samples}
    if len(reviews) < 2:
        raise ValueError("At least two original review sets are required")
    votes = {ident: [] for ident in ids}
    for review in reviews:
        if not isinstance(review, dict) or not set(review) <= ids or any(type(value) is not bool for value in review.values()):
            raise ValueError("Invalid original review")
        for ident, value in review.items():
            votes[ident].append(value)
    if any(not values for values in votes.values()):
        raise ValueError("Every image must have an original review")
    if not isinstance(predictions, dict) or set(predictions) != ids or any(v is not None and type(v) is not bool for v in predictions.values()):
        raise ValueError("Predictions must exactly match manifest IDs")
    # The observed protocol selected shared-review disagreements and accepted
    # checker/reviewer discrepancies for follow-up, after viewing outcomes.
    expected = {ident for ident, values in votes.items()
                if len(set(values)) > 1 or (predictions[ident] is not None and
                   any(value != predictions[ident] for value in values))}
    chosen = {}
    for row in followup:
        if not isinstance(row, dict) or set(row) != {"sample_id", "decision", "notes"}:
            raise ValueError("Follow-up CSV requires sample_id,decision,notes")
        ident, raw = row["sample_id"], row["decision"]
        if ident in chosen or ident not in expected or not isinstance(raw, str):
            raise ValueError("Duplicate or unexpected follow-up ID")
        text = raw.strip().lower()
        if text == "yes" or text.startswith("yes but "):
            chosen[ident] = True
        elif text == "no" or text.startswith("no but "):
            chosen[ident] = False
        else:
            raise ValueError(f"Unresolved or unrecognized follow-up decision: {ident}")
    if set(chosen) != expected:
        raise ValueError("Follow-up decisions must cover exactly the selected cases")
    labels = {ident: chosen[ident] if ident in chosen else values[0] for ident, values in votes.items()}
    result = score(manifest, labels, predictions)
    result["followup_ids"] = sorted(chosen)
    result["label_note"] = ("Original unanimous or single-reviewer answers plus selected post-hoc "
                            "follow-up answers; not independent validation or a consensus guarantee.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        for index, path in enumerate(args.review):
            if any(path.resolve() == other.resolve() or
                   (path.exists() and other.exists() and path.samefile(other))
                   for other in args.review[:index]):
                raise ValueError("Each original review must be a distinct file")
        def load(path):
            return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        with args.followup.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["sample_id", "decision", "notes"]:
                raise ValueError("Follow-up CSV requires sample_id,decision,notes")
            followup = list(reader)
        result = combine(load(args.manifest), [read_reviews(path) for path in args.review],
                         load(args.predictions), followup)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
