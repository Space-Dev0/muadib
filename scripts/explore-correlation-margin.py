"""Explore a larger correlation margin for fixed multi-matrix endpoints."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from exposure.design_breadth import key, predicate_vocabulary, rule_log_breadth
from exposure.design_diagnostics import diagnose
from exposure.measure import load_rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adjust-reference', action='store_true')
    args = parser.parse_args()
    source = Path('outputs/design-breadth-multimatrix-82531.json').read_bytes()
    frozen = json.loads(source)
    report_paths = ('outputs/exposure-57-fresh-82505.json', 'outputs/exposure-57-fresh-82501.json',
                    'outputs/exposure-57-fresh-82506.json')
    breadth_paths = ('outputs/breadth-57-fresh-82505.json', 'outputs/breadth-57-fresh-82501.json',
                     'outputs/breadth-57-fresh-82506.json')
    report_raw = [Path(x).read_bytes() for x in report_paths]
    breadth_raw = [Path(x).read_bytes() for x in breadth_paths]
    reports = [json.loads(x) for x in report_raw]; breadths = [json.loads(x) for x in breadth_raw]
    if any(hashlib.sha256(r).hexdigest() != b['source_sha256'] for r,b in zip(report_raw,breadths,strict=True)):
        raise ValueError('Breadth source hash mismatch')
    rules = load_rules(reports[0]['rules']); names = list(rules); predicates = predicate_vocabulary(rules, 'role')
    designs = frozen['designs']
    spec = importlib.util.spec_from_file_location('breadth_exploration', Path(__file__).with_name('explore-breadth-design.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    moments = []
    for report, breadth in zip(reports, breadths, strict=True):
        if list(load_rules(report['rules'])) != names or breadth['predicate_keys'] != [key(x) for x in predicates]:
            raise ValueError('Rule or predicate mismatch')
        diag = diagnose(report, {'designs': designs})
        if diag['excluded_count']:
            raise ValueError('Excluded rows')
        array = np.asarray(breadth['counts_by_condition'], dtype=float) / breadth['samples_per_condition']
        by_design = {d['name']: rule_log_breadth(array, [d['weights'][n] for n in names], rules, predicates) for d in designs}
        rows = [{**row, 'log_breadth': by_design[row['design']][row['rule']],
                 'reference_group': float(row['design'].startswith('fitted_'))} for row in diag['rows']]
        if any(row['log_breadth'] is None for row in rows):
            raise ValueError('Undefined breadth')
        if args.adjust_reference:
            features = (*mod.FEATURES, 'log_breadth', 'reference_group')
            groups = {name: [] for name in names}
            for row in rows:
                groups[row['rule']].append([row[f] for f in features])
            arrays = [np.asarray(groups[name], dtype=float) for name in names]
            moments.append((np.stack([a.mean(axis=0) for a in arrays]),
                            np.stack([a.T @ a / len(a) for a in arrays])))
        else:
            moments.append(mod.breadth_moments(rows, names))
    def statistics(moment, weights):
        if not args.adjust_reference:
            return mod.weighted_moment_statistics(*moment, weights)
        means, seconds = moment
        weights = np.asarray(weights) / np.sum(weights)
        mean = weights @ means
        covariance = np.einsum('i,ijk->jk', weights, seconds) - np.outer(mean, mean)
        scale = np.sqrt(np.diag(covariance))
        if not np.isfinite(scale).all() or np.any(scale < 1e-12):
            raise ValueError('Invalid reference-adjusted covariance')
        correlation = covariance / np.outer(scale, scale)
        if np.linalg.eigvalsh(correlation).min() <= 1e-10:
            raise ValueError('Collinear reference-adjusted features')
        return {'marginal_lift_r': float(correlation[0,1]),
                'vif': dict(zip((*mod.FEATURES, 'log_breadth', 'reference_group'),
                                map(float, np.diag(np.linalg.inv(correlation))), strict=True))}
    def objective(weights):
        try:
            result = [statistics(m, weights) for m in moments[:2]]
            return sum(float(np.sum(np.maximum(np.array(list(x['vif'].values()))-1.85, 0)**2))
                       + 25 * max(abs(x['marginal_lift_r'])-.24, 0)**2 for x in result)
        except (ValueError, np.linalg.LinAlgError):
            return 1e6
    initial = np.array([frozen['rule_allocation'][name] for name in names])
    opt = minimize(objective, initial, method='SLSQP', bounds=[(.05/len(names), 4/len(names))]*len(names),
                   constraints={'type':'eq', 'fun':lambda x: x.sum()-1}, options={'maxiter':1000,'ftol':1e-10})
    if not opt.success or abs(opt.x.sum()-1)>1e-8:
        raise ValueError(f'Optimizer failed: {opt.message}')
    result = {'schema_version':1, 'source_sha256':frozen['source_sha256'], 'rule_order':names,
              'designs':designs, 'allocations':{'wider_five_feature':dict(zip(names,map(float,opt.x),strict=True))},
              'frozen_endpoints_sha256':hashlib.sha256(source).hexdigest(),
              'report_sha256':[hashlib.sha256(x).hexdigest() for x in report_raw],
              'objective':float(opt.fun), 'effective_rules':float(1/np.sum(opt.x**2)),
              'adjust_reference':args.adjust_reference,
              'statistics':[statistics(m,opt.x) for m in moments],
              'note':'Exploratory local fit to two estimated matrices; third was previously inspected. Not a gate.'}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:result[k] for k in ('objective','effective_rules','statistics')},indent=2))


if __name__ == '__main__':
    main()
