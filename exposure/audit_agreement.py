"""Describe agreement between independently supplied human audit reviews.

This utility does not assign a gold label, resolve disagreements, or decide
whether a checker or research gate passes.
"""

import argparse
import itertools
import json
from pathlib import Path

from exposure.human_audit import _json, _protect_output, _samples, _write_json, read_reviews


def summarize(manifest: object, reviews: dict[str, dict[str, bool]]) -> dict:
    """Compare complete binary review sets, preserving unresolved disagreements."""
    samples = _samples(manifest)
    identifiers = {sample["sample_id"] for sample in samples}
    if not isinstance(reviews, dict) or len(reviews) < 2:
        raise ValueError("At least two named reviewers are required")
    for name, judgments in reviews.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Reviewer names must be nonempty strings")
        if (not isinstance(judgments, dict) or set(judgments) != identifiers or
                any(type(value) is not bool for value in judgments.values())):
            raise ValueError(f"Reviewer {name!r} must have a binary judgment for every manifest ID")

    pairs = []
    for first, second in itertools.combinations(sorted(reviews), 2):
        counts = {"yes_yes": 0, "yes_no": 0, "no_yes": 0, "no_no": 0}
        for identifier in sorted(identifiers):
            key = ("yes" if reviews[first][identifier] else "no") + "_" + (
                "yes" if reviews[second][identifier] else "no")
            counts[key] += 1
        agree = counts["yes_yes"] + counts["no_no"]
        pairs.append({"reviewer_a": first, "reviewer_b": second, "counts": counts,
                      "agreement_fraction": agree / len(identifiers)})

    disagreements = []
    for identifier in sorted(identifiers):
        judgments = {name: reviews[name][identifier] for name in sorted(reviews)}
        if len(set(judgments.values())) > 1:
            disagreements.append({"sample_id": identifier, "judgments": judgments})
    return {"schema_version": 1, "sample_count": len(identifiers),
            "reviewers": sorted(reviews), "pairs": pairs,
            "unanimous_count": len(identifiers) - len(disagreements),
            "disagreements": disagreements,
            "note": ("Descriptive agreement only; no consensus labels, checker accuracy, "
                     "independence, uncertainty interval, or Stage 0 gate is established.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True,
                        help="one CSV per independently completed reviewer; filename is reviewer ID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if len({path.stem for path in args.review}) != len(args.review):
            raise ValueError("Review filenames must have distinct stems")
        _protect_output(args.output, [args.manifest, *args.review])
        reviews = {path.stem: read_reviews(path) for path in args.review}
        _write_json(args.output, summarize(_json(args.manifest), reviews))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
