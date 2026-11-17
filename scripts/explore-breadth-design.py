"""Add exploratory role-sensitive breadth to frozen two-reference designs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from exposure.design_breadth import key, rule_log_breadth
from exposure.design_diagnostics import FEATURES, diagnose
from exposure.design_extremes import generate
from exposure.design_reference_explore import matrix, optimize_reference
from exposure.design_rule_allocation import fit, moments
from exposure.measure import load_rules


def full_statistics(rows, weights):
    features = (*FEATURES, 'log_breadth')
    values = np.asarray([[r[f] for f in features] for r in rows], dtype=float)
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    if not np.isfinite(values).all() or np.any(w < 0):
        raise ValueError('Nonfinite feature or negative allocation')
    centered = values - w @ values
    cov = (centered * w[:, None]).T @ centered
    sd = np.sqrt(np.diag(cov))
    if np.any(sd < 1e-12):
        raise ValueError('A feature is constant')
    corr = cov / np.outer(sd, sd)
    if np.linalg.eigvalsh(corr).min() <= 1e-10:
        raise ValueError('Features are collinear or numerically unstable')
    return {'marginal_lift_r': float(corr[0, 1]),
            'vif': dict(zip(features, map(float, np.diag(np.linalg.inv(corr))), strict=True))}


def breadth_moments(rows, names):
    features = (*FEATURES, 'log_breadth')
    groups = {name: [] for name in names}
    for row in rows:
        groups[row['rule']].append([row[f] for f in features])
    if len({len(group) for group in groups.values()}) != 1 or not all(groups.values()):
        raise ValueError('Expected complete rule groups')
    arrays = [np.asarray(groups[name], dtype=float) for name in names]
    return (np.stack([a.mean(axis=0) for a in arrays]),
            np.stack([a.T @ a / len(a) for a in arrays]))


def weighted_moment_statistics(means, seconds, weights):
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    mean = w @ means
    cov = np.einsum('i,ijk->jk', w, seconds) - np.outer(mean, mean)
    sd = np.sqrt(np.diag(cov))
    if not np.isfinite(sd).all() or np.any(sd < 1e-12):
        raise ValueError('Invalid covariance')
    corr = cov / np.outer(sd, sd)
    if np.linalg.eigvalsh(corr).min() <= 1e-10:
        raise ValueError('Collinear features')
    return {'marginal_lift_r': float(corr[0, 1]),
            'vif': dict(zip((*FEATURES, 'log_breadth'), map(float, np.diag(np.linalg.inv(corr))), strict=True))}


def fit_five(means, seconds, initial, *, min_multiple=0.2, max_multiple=2.0):
    n = len(initial)
    def objective(w):
        try:
            result = weighted_moment_statistics(means, seconds, w)
            vif = np.asarray(list(result['vif'].values()))
            return float(np.sum(np.maximum(vif - 1.75, 0) ** 2)
                         + max(abs(result['marginal_lift_r']) - 0.25, 0) ** 2)
        except (ValueError, np.linalg.LinAlgError):
            return 1e6
    opt = minimize(objective, initial, method='SLSQP', bounds=[(min_multiple/n, max_multiple/n)] * n,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                   options={'maxiter': 1000, 'ftol': 1e-11})
    if not opt.success or abs(opt.x.sum() - 1) > 1e-8:
        raise ValueError(f'Five-feature allocation failed: {opt.message}')
    return opt.x, float(opt.fun)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new machine-readable result path')
    parser.add_argument('--training-report', default='outputs/exposure-57-pilot.json')
    parser.add_argument('--holdout-report', default='outputs/exposure-57-holdout-81205.json')
    parser.add_argument('--training-breadth', default='outputs/breadth-57-pilot-82500.json')
    parser.add_argument('--holdout-breadth', default='outputs/breadth-57-holdout-82500.json')
    args = parser.parse_args()
    source_paths = (args.training_report, args.holdout_report)
    source_raw = [Path(path).read_bytes() for path in source_paths]
    reports = [json.loads(raw) for raw in source_raw]
    breadth_paths = (args.training_breadth, args.holdout_breadth)
    breadth_raw = [Path(path).read_bytes() for path in breadth_paths]
    breadth = [json.loads(raw) for raw in breadth_raw]
    names, values, counts = matrix(reports[0]); rules = load_rules(reports[0]['rules'])
    predicates = sorted({p for atoms in rules.values() for p in atoms}, key=key)
    designs = []
    references = [('uniform', dict.fromkeys(names, 1 / len(names))),
                  ('fitted', dict(zip(names, map(float, optimize_reference(values, counts)), strict=True)))]
    for prefix, ref in references:
        for design in generate(reports[0], ref, min_weight=0.0001)['designs']:
            designs.append({'name': prefix + '_' + design['name'], 'weights': design['weights']})
    diagnoses = [diagnose(report, {'designs': designs}) for report in reports]
    if any(result['excluded_count'] for result in diagnoses):
        raise ValueError('Incomplete design rows')
    q, objective = fit(*moments(diagnoses[0]['rows'], names))
    full_moments = []
    baseline_statistics = {}
    for label, report, raw, b, diag in zip(('pilot', 'holdout'), reports, source_raw, breadth, diagnoses, strict=True):
        if b['source_sha256'] != hashlib.sha256(raw).hexdigest():
            raise ValueError('Breadth source hash does not match report')
        if b['rule_order'] != names or b['predicate_keys'] != [key(p) for p in predicates]:
            raise ValueError('Breadth data do not match ordered rules')
        if b['source_seed'] != report['seed'] or b['samples_per_condition'] != report['samples_per_condition']:
            raise ValueError('Breadth sampling does not match report')
        array = np.asarray(b['counts_by_condition'], dtype=float) / b['samples_per_condition']
        b_by_design = {}
        for d in designs:
            w = [d['weights'][name] for name in names]
            b_by_design[d['name']] = rule_log_breadth(array, w, rules, predicates)
        rows = []
        for row in diag['rows']:
            value = b_by_design[row['design']][row['rule']]
            if value is None:
                raise ValueError('Undefined breadth')
            rows.append({**row, 'log_breadth': value})
        allocated = [q[names.index(r['rule'])] for r in rows]
        baseline_statistics[label] = full_statistics(rows, allocated)
        print(label, json.dumps(baseline_statistics[label], sort_keys=True),
              'rows', len(rows), 'original_fit_objective', objective)
        full_moments.append(breadth_moments(rows, names))
    new_weights, new_objective = fit_five(*full_moments[0], q)
    for label, values in zip(('pilot', 'holdout'), full_moments, strict=True):
        print(label, 'five_feature_allocation', json.dumps(weighted_moment_statistics(*values, new_weights),
                                                          sort_keys=True), 'objective', new_objective,
              'effective_rules', float(1 / np.sum(new_weights ** 2)))
    wider, wider_objective = fit_five(*full_moments[0], new_weights, min_multiple=0.05, max_multiple=4.0)
    for label, values in zip(('pilot', 'holdout'), full_moments, strict=True):
        print(label, 'wider_allocation', json.dumps(weighted_moment_statistics(*values, wider), sort_keys=True),
              'objective', wider_objective, 'effective_rules', float(1 / np.sum(wider ** 2)),
              'min_weight_multiple', float(wider.min() * len(wider)),
              'max_weight_multiple', float(wider.max() * len(wider)))
    if args.output:
        result = {'schema_version': 1,
                  'source_sha256': [hashlib.sha256(raw).hexdigest() for raw in source_raw],
                  'breadth_sha256': [hashlib.sha256(raw).hexdigest() for raw in breadth_raw],
                  'rule_order': names, 'designs': designs,
                  'allocations': {'four_feature': dict(zip(names, map(float, q), strict=True)),
                                  'five_feature': dict(zip(names, map(float, new_weights), strict=True)),
                                  'wider_five_feature': dict(zip(names, map(float, wider), strict=True))},
                  'baseline_statistics': baseline_statistics,
                  'five_feature_statistics': {label: weighted_moment_statistics(*value, new_weights)
                                              for label, value in zip(('pilot', 'holdout'), full_moments, strict=True)},
                  'wider_statistics': {label: weighted_moment_statistics(*value, wider)
                                       for label, value in zip(('pilot', 'holdout'), full_moments, strict=True)},
                  'bounds_relative_to_equal': {'five_feature': [0.2, 2.0], 'wider': [0.05, 4.0]},
                  'note': 'Exploratory, post-hoc allocations and role-sensitive breadth; not a Stage 0 gate.'}
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    main()
