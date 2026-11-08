"""Multi-chain hit-and-run sampling with convergence diagnostics.

A single chain cannot distinguish a narrow design space from slow mixing.
This module runs independent chains from the same reference mixture, reports
split-R-hat and autocorrelation ESS for each rule's log-lift series, and keeps
the realized per-rule range.  The thresholds are diagnostics, not a Stage 0
gate; small exploratory runs are expected to fail them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from exposure.design_sample import sample_designs
from exposure.measure import load_rules, unique_object
from exposure.mixture import summarize

RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 100.0


def split_rhat(chains: list[np.ndarray]) -> float | None:
    """Return the split-R-hat statistic, or None when it is undefined."""
    arrays = [np.asarray(values, dtype=float) for values in chains]
    if len(arrays) < 2 or any(array.ndim != 1 for array in arrays):
        return None
    length = min(len(array) for array in arrays)
    if length < 4:
        return None
    half = length // 2
    halves = []
    for array in arrays:
        halves.append(array[:half])
        halves.append(array[half:2 * half])
    matrix = np.asarray(halves, dtype=float)
    count, draws = matrix.shape
    if count < 2 or draws < 2:
        return None
    means = matrix.mean(axis=1)
    variances = matrix.var(axis=1, ddof=1)
    within = float(variances.mean())
    between = float(draws * means.var(ddof=1))
    if not math.isfinite(within) or within <= 0.0:
        return None
    variance = ((draws - 1) / draws) * within + between / draws
    if variance <= 0.0:
        return None
    return math.sqrt(variance / within)


def effective_sample_size(values: np.ndarray) -> float | None:
    """Return an autocorrelation ESS using Geyer's initial positive sequence."""
    sequence = np.asarray(values, dtype=float)
    count = len(sequence)
    if sequence.ndim != 1 or count < 4 or not np.isfinite(sequence).all():
        return None
    centered = sequence - sequence.mean()
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        return None
    autocovariance = np.correlate(centered, centered, mode="full")[count - 1:] / denominator
    total = 0.0
    for lag in range(0, count - 1, 2):
        pair = autocovariance[lag] + (autocovariance[lag + 1]
                                      if lag + 1 < count else 0.0)
        if pair <= 0.0:
            break
        total += pair
    tau = -1.0 + 2.0 * total
    value = count / tau if tau > 0.0 else float(count)
    return max(1.0, min(float(count), value))


def sample_chains(report: dict, weights: dict, *, chains: int, samples: int,
                  seed: int, min_weight: float = 0.0, burn_in: int = 100,
                  thin: int = 10) -> dict:
    """Run independent chains and report convergence diagnostics on rule lifts."""
    if type(chains) is not int or chains < 1:
        raise ValueError("chains must be a positive integer")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    rules = load_rules(report.get("rules"))
    names = list(rules)
    chain_results = []
    series: dict[str, list[list[float]]] = {name: [[] for _ in range(chains)]
                                            for name in names}
    for index in range(chains):
        result = sample_designs(report, weights, samples=samples, seed=seed + index,
                                min_weight=min_weight, burn_in=burn_in, thin=thin)
        chain_results.append({"seed": seed + index, "designs": result["designs"]})
        for design in result["designs"]:
            summary = summarize(report, design["weights"])
            for name in names:
                value = summary["results"][name]["log_lift"]
                series[name][index].append(float("nan") if value is None else float(value))

    diagnostics = {}
    for name, chain_values in series.items():
        arrays = [np.asarray(values, dtype=float) for values in chain_values]
        finite = all(np.all(np.isfinite(array)) for array in arrays)
        rhat = split_rhat(arrays) if finite else None
        ess_values = [effective_sample_size(array) for array in arrays
                      if np.all(np.isfinite(array))]
        ess_values = [value for value in ess_values if value is not None]
        pooled = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
        means = [float(array.mean()) for array in arrays
                 if len(array) and np.all(np.isfinite(array))]
        diagnostics[name] = {
            "rhat": rhat,
            "min_chain_ess": min(ess_values) if ess_values else None,
            "median_chain_ess": float(np.median(ess_values)) if ess_values else None,
            "per_chain_mean_range": (max(means) - min(means)) if len(means) > 1 else None,
            "range": (float(pooled.max() - pooled.min()) if finite and pooled.size else None),
            "finite": finite,
        }

    rhats = [entry["rhat"] for entry in diagnostics.values()
             if entry["rhat"] is not None]
    esses = [entry["min_chain_ess"] for entry in diagnostics.values()
             if entry["min_chain_ess"] is not None]
    ranges = [entry["range"] for entry in diagnostics.values()
              if entry["range"] is not None]
    if rhats and max(rhats) > RHAT_THRESHOLD:
        status = "fail"
    elif esses and min(esses) < ESS_THRESHOLD:
        status = "fail"
    elif len(rhats) != len(diagnostics) or len(esses) != len(diagnostics):
        status = "insufficient"
    else:
        status = "pass"
    summary = {
        "rule_count": len(names),
        "max_rhat": max(rhats) if rhats else None,
        "min_ess": min(esses) if esses else None,
        "median_rule_range": float(np.median(ranges)) if ranges else None,
        "max_rule_range": max(ranges) if ranges else None,
        "status": status,
        "thresholds": {"rhat": RHAT_THRESHOLD, "ess": ESS_THRESHOLD},
    }
    return {
        "schema_version": 1,
        "chains": chains,
        "samples_per_chain": samples,
        "burn_in": burn_in,
        "thin": thin,
        "min_weight": min_weight,
        "source_seed": report.get("seed"),
        "chain_results": chain_results,
        "diagnostics": {"summary": summary, "rules": diagnostics},
        "note": ("Split-R-hat and autocorrelation ESS are diagnostics, not a Stage 0 gate. "
                 "A narrow realized range with poor diagnostics indicates slow mixing, not "
                 "an infeasible design space."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-chain hit-and-run designs with convergence diagnostics")
    parser.add_argument("--report", type=Path, required=True,
                        help="subset-enabled measurement JSON")
    parser.add_argument("--weights", type=Path, required=True,
                        help="reference condition-weight JSON")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=100,
                        help="samples per chain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/design-chains.json"))
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"),
                            object_pairs_hook=unique_object)
        weights = json.loads(args.weights.read_text(encoding="utf-8"),
                             object_pairs_hook=unique_object)
        result = sample_chains(report, weights, chains=args.chains, samples=args.samples,
                               seed=args.seed, min_weight=args.min_weight,
                               burn_in=args.burn_in, thin=args.thin)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
