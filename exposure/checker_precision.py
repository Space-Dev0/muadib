"""Exact binomial planning bounds for checker decisions in one class.

These bounds require independent, identically distributed evaluated units.
They do not turn convenience samples or repeated reviewer labels into new units.
"""
from __future__ import annotations

import argparse
import json
import math

from scipy.stats import beta


def upper_error_bound(errors: int, accepted: int, alpha: float = .05) -> float:
    """One-sided Clopper–Pearson upper bound for error among accepted cases."""
    if (type(errors) is not int or type(accepted) is not int or accepted < 1
            or not 0 <= errors <= accepted or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("Require 0 <= errors <= accepted, accepted > 0, and 0 < alpha < 1")
    return 1.0 if errors == accepted else float(beta.ppf(1 - alpha, errors + 1, accepted - errors))


def zero_error_sample_size(target: float, alpha: float = .05) -> int:
    """Accepted independent cases needed for a zero-error upper bound <= target."""
    if not all(math.isfinite(x) and 0 < x < 1 for x in (target, alpha)):
        raise ValueError("target and alpha must lie strictly between zero and one")
    n = math.ceil(math.log(alpha) / math.log1p(-target))
    # Correct possible floating-point rounding at the integer boundary.
    while 1 - alpha ** (1 / n) > target:
        n += 1
    while n > 1 and 1 - alpha ** (1 / (n - 1)) <= target:
        n -= 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=.02)
    parser.add_argument("--family-alpha", type=float, default=.05)
    parser.add_argument("--classes", type=int, default=2,
                        help="Bonferroni-adjusted class-specific error claims")
    args = parser.parse_args()
    if args.classes < 1 or not math.isfinite(args.family_alpha) or not 0 < args.family_alpha < 1:
        parser.error("classes must be positive and family alpha must lie between zero and one")
    try:
        alpha = args.family_alpha / args.classes
        n = zero_error_sample_size(args.target, alpha)
        print(json.dumps({"target": args.target, "family_alpha": args.family_alpha,
                          "classes": args.classes, "per_class_alpha": alpha,
                          "accepted_cases_per_class_if_zero_errors": n,
                          "upper_bound_at_zero_errors": upper_error_bound(0, n, alpha),
                          "note": "Conditional on accepted cases and iid units; coverage, abstentions, "
                                  "human ambiguity and model-seed variability require separate analysis."}, indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
