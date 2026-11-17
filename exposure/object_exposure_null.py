"""Independent-object scene baseline for same-object conjunctions.

Assumes independent and identically distributed objects, mutually independent
attribute axes within each object, and a specified object-count distribution.
This is an illustrative null, not the existing renderer's validated null.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be a finite probability in [0,1]')
    return float(value)


def _at_least_one(p, count):
    if count == 0 or p == 0: return 0.0
    if p == 1: return 1.0
    return -math.expm1(count * math.log1p(-p))


def independent_object_null(probabilities: Sequence[float], count_distribution: Mapping[int, float],
                            *, observed_joint: float | None = None) -> dict:
    """Calculate scene singleton/joint probabilities for an explicit K mixture.

    `probabilities` are same-object attribute probabilities, not scene-level
    frequencies. Object count K may vary; each weight is P(K). Observed joint,
    when supplied, must refer to the same scene population and bound event.
    """
    if not probabilities: raise ValueError('at least one attribute is required')
    p = [_probability(value, f'probabilities[{i}]') for i, value in enumerate(probabilities)]
    if not isinstance(count_distribution, Mapping) or not count_distribution:
        raise ValueError('count distribution must be nonempty')
    counts=[]
    for k, weight in count_distribution.items():
        if type(k) is not int or k < 0: raise ValueError('object counts must be nonnegative integers')
        counts.append((k, _probability(weight, f'weight[{k}]')))
    if not math.isclose(math.fsum(w for _,w in counts), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError('count probabilities must sum to one')
    obj_joint = math.prod(p)
    singleton = [math.fsum(w * _at_least_one(value, k) for k,w in counts) for value in p]
    baseline = math.fsum(w * _at_least_one(obj_joint, k) for k,w in counts)
    product = math.prod(singleton)
    scene_lift = math.log(baseline / product) if baseline > 0 and product > 0 else None
    observed = None if observed_joint is None else _probability(observed_joint, 'observed_joint')
    adjusted = math.log(observed / baseline) if observed is not None and observed > 0 and baseline > 0 else None
    return {'object_probabilities':p, 'count_distribution':{str(k):w for k,w in sorted(counts)},
            'object_joint_independent':obj_joint, 'scene_singletons':singleton,
            'scene_joint_independent':baseline, 'scene_singleton_product':product,
            'scene_log_lift_under_null':scene_lift, 'observed_joint':observed,
            'observed_log_ratio_to_independent_object_null':adjusted,
            'assumptions':'iid objects, independent attribute axes, and count distribution independent of attributes; not a fitted renderer null'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probabilities', nargs='+', type=float, required=True)
    p.add_argument('--counts', required=True, help='JSON object mapping integer object counts to probabilities')
    p.add_argument('--observed-joint', type=float)
    args=p.parse_args()
    try:
        raw=json.loads(args.counts)
        if not isinstance(raw, dict) or any(not isinstance(k,str) or not k.isdigit() or str(int(k)) != k for k in raw):
            raise ValueError('count keys must be canonical nonnegative integers')
        result=independent_object_null(args.probabilities, {int(k):v for k,v in raw.items()}, observed_joint=args.observed_joint)
    except (ValueError,TypeError) as exc: p.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__': main()
