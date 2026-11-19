"""Simultaneous uncertainty bounds for a frozen four-mixture design.

The design must be fixed independently of the evaluated holdout. Bounds use
weighted Hoeffding inequalities and a union bound; no independence between
predicate events in the same scene is assumed. This does not establish the
remaining Stage 0 gates or account for dataset/model variation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from exposure.design_factorial import contrasts, load_matrix


def interval_audit(weights, marg, joint, sample_sizes, *, delta_m, delta_l,
                   cross_tolerance, alpha=0.05):
    """Bound all marginal/lift contrast margins simultaneously.

    There are four mixture means per rule for each of two singletons and the
    conjunction. The union bound covers all such means, even though they are
    dependent. Each condition must contain independent accepted scenes, with
    independent sampling across conditions. Weights and target rules must be
    fixed independently of the report being evaluated.
    """
    w, m, j, n = (np.asarray(x, dtype=float) for x in (weights, marg, joint, sample_sizes))
    if j.ndim != 2:
        raise ValueError('joint matrix must have rules by conditions')
    rules, conditions = j.shape
    if (rules < 1 or conditions < 1 or w.shape != (4, conditions)
            or m.shape != (rules, 2, conditions) or n.shape != (conditions,)):
        raise ValueError('incompatible matrix shapes')
    if (not all(np.isfinite(x).all() for x in (w, m, j, n))
            or (w < 0).any() or not np.allclose(w.sum(axis=1), 1, rtol=0, atol=1e-8)
            or (n < 1).any() or not np.equal(n, np.floor(n)).all()
            or (m < 0).any() or (m > 1).any() or (j < 0).any() or (j > 1).any()
            or not math.isfinite(alpha) or not 0 < alpha < 1
            or not all(math.isfinite(x) and x >= 0 for x in (delta_m, delta_l, cross_tolerance))):
        raise ValueError('invalid probabilities, weights, sample sizes, or settings')

    mean_m = np.einsum('rpj,kj->krp', m, w)
    mean_j = np.einsum('rj,kj->kr', j, w)
    if (mean_m <= 0).any() or (mean_j <= 0).any():
        raise ValueError('a point-estimate exposure is zero; log features are undefined')
    tests = 4 * rules * 3
    radius = np.sqrt(np.log(2 * tests / alpha) / 2 * (w ** 2 / n).sum(axis=1))
    lower_m = np.maximum(0, mean_m - radius[:, None, None])
    upper_m = np.minimum(1, mean_m + radius[:, None, None])
    lower_j = np.maximum(0, mean_j - radius[:, None])
    upper_j = np.minimum(1, mean_j + radius[:, None])
    with np.errstate(divide='ignore'):
        lo = np.stack((np.log(lower_m).mean(axis=2),
                       np.log(lower_j) - np.log(upper_m).sum(axis=2)), axis=2)
        hi = np.stack((np.log(upper_m).mean(axis=2),
                       np.log(upper_j) - np.log(lower_m).sum(axis=2)), axis=2)
    point = contrasts(w.ravel(), m, j, delta_m, delta_l, cross_tolerance)[0]
    bounds = []
    for high, low, feature, delta in ((2, 0, 0, delta_m), (3, 1, 0, delta_m),
                                      (1, 0, 1, delta_l), (3, 2, 1, delta_l)):
        for r in range(rules):
            bounds.append((lo[high, r, feature] - hi[low, r, feature] - delta,
                           hi[high, r, feature] - lo[low, r, feature] - delta))
    for high, low, feature in ((1, 0, 0), (3, 2, 0), (2, 0, 1), (3, 1, 1)):
        for r in range(rules):
            low_diff = lo[high, r, feature] - hi[low, r, feature]
            high_diff = hi[high, r, feature] - lo[low, r, feature]
            bounds.extend(((cross_tolerance - high_diff, cross_tolerance - low_diff),
                           (cross_tolerance + low_diff, cross_tolerance + high_diff)))
    rows = []
    for estimate, (low, high) in zip(point, bounds, strict=True):
        status = 'certified_met' if low >= 0 else ('certified_violated' if high < 0 else 'unresolved')
        rows.append({'estimate': float(estimate), 'lower': float(low) if math.isfinite(low) else None,
                     'upper': float(high) if math.isfinite(high) else None, 'status': status})
    return {'alpha': alpha, 'family_size': tests, 'mixture_radii': radius.tolist(),
            'constraints': rows, 'counts': {s: sum(row['status'] == s for row in rows)
                                           for s in ('certified_met', 'certified_violated', 'unresolved')},
            'note': 'Conservative simultaneous bounds conditional on an independent, frozen design; '
                    'null endpoints denote unbounded log intervals. Not a Stage 0 decision.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True,
                        help='Original fitting report, to verify unchanged rule definitions')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=.05)
    args = parser.parse_args()
    design_raw, report_raw = args.design.read_bytes(), args.report.read_bytes()
    reference_raw = args.reference_report.read_bytes()
    design, report, reference = json.loads(design_raw), json.loads(report_raw), json.loads(reference_raw)
    if not design.get('best') or not design['best'].get('feasible'):
        raise ValueError('design has no fitted solution')
    if (hashlib.sha256(reference_raw).hexdigest() != design.get('report_sha256')
            or report.get('rules') != reference.get('rules')
            or report.get('rule_order') != reference.get('rule_order')
            or not reference.get('estimator_sha256')
            or report.get('estimator_sha256') != reference.get('estimator_sha256')
            or report.get('estimator') != reference.get('estimator')):
        raise ValueError('fitting and holdout definitions, estimator or condition order differ')
    if (not isinstance(design.get('fitting_seeds'), list)
            or not design['fitting_seeds'] or reference.get('seed') not in design['fitting_seeds']):
        raise ValueError('design is missing fitting-seed provenance')
    if (report.get('seed') is None or report.get('seed') in design.get('fitting_seeds', [])
            or hashlib.sha256(report_raw).hexdigest() in design.get('fitting_report_sha256', [])):
        raise ValueError('report must have a seed distinct from fitting reports')
    order, marg, joint = load_matrix(report, design['rule_names'])
    if order != design['condition_order']:
        raise ValueError('condition order mismatch')
    by_condition = {col['condition']: col for col in report['columns']}
    sizes = [by_condition[name]['accepted_scenes'] for name in order]
    for name, size in zip(order, sizes, strict=True):
        if type(size) is not int or size < 1:
            raise ValueError('invalid accepted scene count')
        col = by_condition[name]
        for rule in design['rule_names']:
            subsets = col['predicate_subsets'][rule]
            selected = [row for row in subsets if row['predicate_indices'] in ([0], [1])]
            if len(selected) != 2 or {tuple(row['predicate_indices']) for row in selected} != {(0,), (1,)}:
                raise ValueError('missing or duplicated singleton measurements')
            for entry in [col['measurements'][rule], *selected]:
                count, probability = entry['successes'], entry['probability']
                if (type(count) is not int or not 0 <= count <= size
                        or not math.isclose(probability, count / size, rel_tol=0, abs_tol=1e-12)):
                    raise ValueError('inconsistent probability, success count, or sample size')
    settings = design['settings']
    result = interval_audit(design['best']['weights'], marg, joint, sizes,
                            delta_m=settings['delta_m'], delta_l=settings['delta_l'],
                            cross_tolerance=settings['cross_tolerance'], alpha=args.alpha)
    result.update({'design_sha256': hashlib.sha256(design_raw).hexdigest(),
                   'report_sha256': hashlib.sha256(report_raw).hexdigest(), 'seed': report['seed'],
                   'rule_names': design['rule_names'],
                   'caution': 'The analyst must verify the design was frozen before this report was generated.'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'seed': result['seed'], 'counts': result['counts'],
                      'mixture_radii': result['mixture_radii']}))


if __name__ == '__main__':
    main()
