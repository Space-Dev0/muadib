"""Composition-gap estimands for the exposure experiment.

The proposal defines the composition gap as

    Gamma_r = logit(RSR_r) - logit(prod_p c_p).

Because ``logit`` is injective, this quantity is exactly zero whenever
``RSR_r == prod_p c_p`` in the interior of the probability domain.  The
additive log-scale gap

    Gamma_log_r = log(RSR_r) - sum_p log(c_p)

is a different estimand.  It is also zero under the multiplicative null, but
unlike ``Gamma_r`` it decomposes into one additive term per predicate.  Both are
reported here so analyses cannot silently substitute one scale for the other.

Boundary values are returned as ``None`` with a status rather than as
infinities: empirical zeros are not evidence that a rule is impossible, and
saturated empirical ones are not evidence that it is certain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

LOGIT_UNDEFINED = "undefined"
LOGIT_DEFINED = "defined"


def _probability(value: object, name: str, *, allow_one: bool = False) -> float:
    """Validate a finite probability, optionally allowing the value one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if not (0.0 < value and upper_ok):
        raise ValueError(f"{name} must lie in {'(0, 1]' if allow_one else '(0, 1)'}")
    return value


def logit(probability: float) -> float:
    """Return ``log(p / (1 - p))`` for ``0 < p < 1``.

    Raises ``ValueError`` at the boundary instead of returning an infinity.
    """
    probability = _probability(probability, "probability")
    return math.log(probability / (1.0 - probability))


def gamma_logit(rsr: float, marginals: Sequence[float]) -> float:
    """Return the proposal's logit-scale composition gap.

    ``marginals`` are the singleton rule-satisfaction rates ``c_p``.  The
    The RSR and marginal product must be strictly inside ``(0, 1)``;
    individual marginals may equal one. Use :func:`composition_gaps` for
    boundary-aware reporting.
    """
    rsr = _probability(rsr, "rsr")
    if not marginals:
        raise ValueError("At least one singleton marginal is required")
    values = [_probability(value, f"marginal[{index}]", allow_one=True)
              for index, value in enumerate(marginals)]
    product = math.prod(values)
    if not 0.0 < product < 1.0:
        raise ValueError("The product of marginals must lie in (0, 1)")
    return logit(rsr) - logit(product)


def gamma_log(rsr: float, marginals: Sequence[float]) -> float:
    """Return the additive log-scale gap ``log RSR - sum log c_p``."""
    rsr = _probability(rsr, "rsr", allow_one=True)
    if not marginals:
        raise ValueError("At least one singleton marginal is required")
    values = [_probability(value, f"marginal[{index}]", allow_one=True)
              for index, value in enumerate(marginals)]
    return math.log(rsr) - math.fsum(math.log(value) for value in values)


def composition_gaps(rsr: float, marginals: Sequence[float]) -> dict:
    """Return both gaps with explicit boundary status.

    ``gamma_logit`` requires the RSR and marginal product inside ``(0, 1)``.
    ``gamma_log`` allows values equal to one because ``log(1) = 0``.  If an
    input is zero, the corresponding gap is ``None`` and the status records
    why; the raw marginal product is still returned.
    """
    if not marginals:
        raise ValueError("At least one singleton marginal is required")
    if isinstance(rsr, bool) or not isinstance(rsr, (int, float)) or not math.isfinite(rsr):
        raise ValueError("rsr must be a finite number")
    for index, value in enumerate(marginals):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"marginal[{index}] must be a finite number")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"marginal[{index}] must lie in [0, 1]")
    symbolic = [float(value) for value in marginals]
    rsr = float(rsr)
    if not 0.0 <= rsr <= 1.0:
        raise ValueError("rsr must lie in [0, 1]")

    product = math.prod(symbolic)
    logit_defined = 0.0 < rsr < 1.0 and 0.0 < product < 1.0
    log_defined = 0.0 < rsr <= 1.0 and all(0.0 < value <= 1.0 for value in symbolic)
    gamma_logit_value = gamma_logit(rsr, symbolic) if logit_defined else None
    gamma_log_value = gamma_log(rsr, symbolic) if log_defined else None
    if logit_defined and log_defined:
        status = "both_defined"
    elif log_defined:
        status = "log_only"
    else:
        status = LOGIT_UNDEFINED
    return {
        "marginal_product": product,
        "gamma_logit": gamma_logit_value,
        "gamma_log": gamma_log_value,
        "gamma_status": status,
    }


def null_gaps(marginals: Sequence[float]) -> dict:
    """Evaluate both gaps at the multiplicative null ``RSR = prod c_p``.

    This is the reproducibility check that separates the proposal's estimand
    from ``sum_p logit(c_p)``: both returned gaps must be zero (up to floating
    point) for every marginal vector inside the interior.
    """
    product = math.prod(marginals)
    gaps = composition_gaps(product, marginals)
    return {"rsr": product, **gaps}


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate composition-gap estimands and their multiplicative null")
    parser.add_argument("--rsr", type=float,
                        help="observed rule-satisfaction rate")
    parser.add_argument("--marginals", type=float, nargs="+", required=True,
                        help="singleton rule-satisfaction rates c_p")
    parser.add_argument("--null", action="store_true",
                        help="ignore --rsr and evaluate at RSR = product(c_p)")
    args = parser.parse_args()
    try:
        if not args.null and args.rsr is None:
            raise ValueError("--rsr is required unless --null is set")
        result = null_gaps(args.marginals) if args.null else composition_gaps(
            args.rsr, args.marginals)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
