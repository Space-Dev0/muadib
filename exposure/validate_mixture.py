"""Compare matrix predictions with independently sampled mixture scenes.

This is an instrument check, not a hypothesis test or a pass/fail gate.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from exposure.measure import load_rules, unique_object, wilson_interval
from exposure.mixture import summarize
from exposure.render import sample_conditioned
from exposure.rules import satisfies


def validate(report: dict, weights: dict, *, samples: int, seed: int,
             max_attempts: int = 10_000) -> dict:
    """Compare A-weighted rule exposures with fresh draws from the mixture.

    The seed must differ from the source seed to avoid identical pseudorandom
    streams. The caller must also ensure that the source report was produced by
    the same renderer version; this cannot be inferred from schema version 1.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    prediction = summarize(report, weights)
    source_seed = report.get("seed")
    if type(source_seed) is not int:
        raise ValueError("Source report must contain an integer seed")
    if seed == source_seed:
        raise ValueError("Validation seed must differ from the measurement report seed")
    rules = load_rules(report["rules"])
    normalized = prediction["weights"]
    names = list(rules)
    rng = random.Random(seed)
    selected = {name: 0 for name in names}
    counts = {name: 0 for name in names}
    total_attempts = 0
    for index in range(samples):
        condition = rng.choices(names, weights=[normalized[name] for name in names], k=1)[0]
        try:
            objects, attempts = sample_conditioned(rng, rules[condition], max_attempts)
        except ValueError as exc:
            raise ValueError(f"Condition {condition!r}, sample {index + 1}: {exc}") from exc
        total_attempts += attempts
        selected[condition] += 1
        for name, predicates in rules.items():
            counts[name] += satisfies(objects, predicates)

    results = {}
    for rule in names:
        predicted = prediction["results"][rule]["rule_exposure"]
        observed = counts[rule] / samples
        results[rule] = {
            "successes": counts[rule],
            "observed_probability": observed,
            "observed_wilson_95": list(wilson_interval(counts[rule], samples)),
            "predicted_probability": predicted,
            "observed_minus_predicted": observed - predicted,
        }
    source_digest = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
    return {
        "schema_version": 1,
        "interpretation": "scene satisfaction in an independently sampled condition mixture",
        "source_report_sha256": source_digest,
        "source_seed": source_seed,
        "source_samples_per_condition": report.get("samples_per_condition"),
        "validation_seed": seed,
        "validation_samples": samples,
        "max_attempts_per_scene": max_attempts,
        "total_attempts": total_attempts,
        "condition_counts": selected,
        "weights": normalized,
        "rules": report["rules"],
        "results": results,
        "note": ("Descriptive only: uncertainty in the matrix prediction and in the difference "
                 "is not quantified; cross-rule dependence, estimated weights and renderer "
                 "changes are not accounted for."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check weighted exposure predictions on fresh scenes")
    parser.add_argument("--report", type=Path, required=True, help="subset-enabled measurement JSON")
    parser.add_argument("--weights", type=Path, required=True, help="condition weight mapping JSON")
    parser.add_argument("--samples", type=int, default=1000, help="fresh accepted scenes")
    parser.add_argument("--seed", type=int, required=True, help="seed distinct from source measurement seed")
    parser.add_argument("--max-attempts", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/mixture-validation.json"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        weights = json.loads(args.weights.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        result = validate(report, weights, samples=args.samples, seed=args.seed, max_attempts=args.max_attempts)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
