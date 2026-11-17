"""Fit a five-feature candidate with singleton equality on two estimated matrices."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from exposure.design_breadth import key, predicate_vocabulary, rule_log_breadth
from exposure.design_diagnostics import diagnose
from exposure.design_multi_matrix import generate
from exposure.design_reference_explore import matrix, optimize_reference
from exposure.design_subspace import diagnose as subspace
from exposure.measure import load_rules


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reports', nargs=3, default=['outputs/exposure-57-fresh-82505.json',
                   'outputs/exposure-57-fresh-82501.json', 'outputs/exposure-57-holdout-81205.json'])
    p.add_argument('--breadths', nargs=3, default=['outputs/breadth-57-fresh-82505.json',
                   'outputs/breadth-57-fresh-82501.json', 'outputs/breadth-57-holdout-82500.json'])
    args = p.parse_args()
    report_paths, breadth_paths = args.reports, args.breadths
    raw = [Path(x).read_bytes() for x in report_paths]
    reports = [json.loads(x) for x in raw]
    breadth_raw = [Path(x).read_bytes() for x in breadth_paths]
    breadths = [json.loads(x) for x in breadth_raw]
    for report_raw, breadth in zip(raw, breadths, strict=True):
        if breadth['source_sha256'] != hashlib.sha256(report_raw).hexdigest():
            raise ValueError('Breadth/source hash mismatch')
    names, values, counts = matrix(reports[0]); rules = load_rules(reports[0]['rules'])
    predicates = predicate_vocabulary(rules, 'role')
    if any(b['predicate_keys'] != [key(x) for x in predicates] for b in breadths):
        raise ValueError('Breadth vocabulary mismatch')
    refs = {'uniform': dict.fromkeys(names, 1 / len(names)),
            'fitted': dict(zip(names, map(float, optimize_reference(values, counts)), strict=True))}
    designs = []
    for prefix, reference in refs.items():
        for d in generate(reports[:2], reference)['designs']:
            designs.append({'name': prefix + '_' + d['name'], 'weights': d['weights']})
    spec = importlib.util.spec_from_file_location('breadth_exploration', Path(__file__).with_name('explore-breadth-design.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows_by_report, moments_by_report, drift = [], [], []
    for report, breadth in zip(reports, breadths, strict=True):
        diag = diagnose(report, {'designs': designs})
        if diag['excluded_count']:
            raise ValueError('Design rows were excluded')
        arr = np.asarray(breadth['counts_by_condition'], dtype=float) / breadth['samples_per_condition']
        bd = {d['name']: rule_log_breadth(arr, [d['weights'][name] for name in names], rules, predicates)
              for d in designs}
        rows = [{**r, 'log_breadth': bd[r['design']][r['rule']]} for r in diag['rows']]
        if any(r['log_breadth'] is None for r in rows):
            raise ValueError('Undefined breadth')
        rows_by_report.append(rows); moments_by_report.append(mod.breadth_moments(rows, names))
        sub = subspace(report, refs['uniform'])
        singleton = np.asarray([x['column_probabilities'] for x in sub['singleton_events']])
        groups = {}
        for prefix, reference in refs.items():
            base = np.asarray([reference[name] for name in names])
            maxima = [float(np.max(np.abs(singleton @ (np.asarray([d['weights'][n] for n in names]) - base))))
                      for d in designs if d['name'].startswith(prefix + '_')]
            groups[prefix] = {'max': max(maxima), 'median_max': float(np.median(maxima))}
        drift.append(groups)
    allocation, objective = mod.fit_five(*moments_by_report[0], np.full(len(names), 1/len(names)),
                                         min_multiple=.05, max_multiple=4.)
    def robust_objective(weights):
        try:
            scores = []
            for moment in moments_by_report[:2]:
                result = mod.weighted_moment_statistics(*moment, weights)
                vifs = np.array(list(result['vif'].values()))
                scores.append(float(np.sum(np.maximum(vifs - 1.75, 0) ** 2)
                                    + max(abs(result['marginal_lift_r']) - .25, 0) ** 2))
            return max(scores)
        except (ValueError, np.linalg.LinAlgError):
            return 1e6
    fit = minimize(robust_objective, allocation, method='SLSQP',
                   bounds=[(.05/len(names), 4./len(names))] * len(names),
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                   options={'maxiter': 1000, 'ftol': 1e-10})
    if not fit.success or abs(fit.x.sum() - 1) > 1e-8:
        raise ValueError(f'Multi-matrix allocation failed: {fit.message}')
    output = {'schema_version': 1, 'source_sha256': [hashlib.sha256(x).hexdigest() for x in raw],
              'breadth_sha256': [hashlib.sha256(x).hexdigest() for x in breadth_raw],
              'seeds': [r['seed'] for r in reports], 'designs': designs,
              'rule_allocation': dict(zip(names, map(float, allocation), strict=True)),
              'robust_rule_allocation': dict(zip(names, map(float, fit.x), strict=True)),
              'objective': objective, 'effective_rules': float(1 / np.sum(allocation ** 2)),
              'robust_objective': float(fit.fun), 'robust_effective_rules': float(1 / np.sum(fit.x ** 2)),
              'statistics': [mod.weighted_moment_statistics(*m, allocation) for m in moments_by_report],
              'robust_statistics': [mod.weighted_moment_statistics(*m, fit.x) for m in moments_by_report],
              'singleton_deviations': drift,
              'note': 'Exploratory multi-matrix equality; third matrix was previously inspected. Not a gate or population guarantee.'}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k: output[k] for k in ('seeds', 'objective', 'effective_rules', 'statistics',
                                             'robust_objective', 'robust_effective_rules',
                                             'robust_statistics', 'singleton_deviations')}, indent=2))


if __name__ == '__main__':
    main()
