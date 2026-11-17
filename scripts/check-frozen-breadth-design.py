"""Evaluate a frozen exploratory allocation on a fresh Monte Carlo matrix."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from exposure.design_breadth import key, predicate_vocabulary, rule_log_breadth, scene_event_key
from exposure.design_diagnostics import diagnose
from exposure.design_subspace import diagnose as subspace
from exposure.measure import load_rules


def reference_adjusted_vifs(rows, row_weights, features):
    names = (*features, 'log_breadth', 'reference_group')
    values = np.asarray([[row[field] for field in names[:-1]]
                         + [float(row['design'].startswith('fitted_'))] for row in rows])
    normalized = np.asarray(row_weights, dtype=float)
    if not np.isfinite(normalized).all() or np.any(normalized < 0) or normalized.sum() <= 0:
        raise ValueError('Invalid row weights')
    normalized = normalized / normalized.sum()
    centered = values - normalized @ values
    covariance = (centered * normalized[:, None]).T @ centered
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale < 1e-12):
        raise ValueError('A reference-adjusted feature is constant')
    correlation = covariance / np.outer(scale, scale)
    if np.linalg.eigvalsh(correlation).min() <= 1e-10:
        raise ValueError('Reference-adjusted features are collinear')
    return dict(zip(names, map(float, np.diag(np.linalg.inv(correlation))), strict=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--breadth', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = [path.read_bytes() for path in (args.frozen, args.report, args.breadth)]
    frozen, report, breadth = [json.loads(data) for data in raw]
    report_hash = hashlib.sha256(raw[1]).hexdigest()
    if report_hash == frozen['source_sha256'][0] or breadth['source_sha256'] != report_hash:
        raise ValueError('Expected a fresh measurement with matching breadth source')
    names = frozen['rule_order']; rules = load_rules(report['rules'])
    if list(rules) != names or breadth['rule_order'] != names:
        raise ValueError('Rule order mismatch')
    mode = breadth.get('mode', 'role')
    predicates = predicate_vocabulary(rules, mode)
    event_key = key if mode == 'role' else scene_event_key
    if breadth['predicate_keys'] != [event_key(predicate) for predicate in predicates]:
        raise ValueError('Predicate vocabulary mismatch')
    if breadth['samples_per_condition'] != report['samples_per_condition'] or breadth['source_seed'] != report['seed']:
        raise ValueError('Breadth sampling mismatch')
    designs = frozen['designs']; diag = diagnose(report, {'designs': designs})
    if diag['excluded_count']:
        raise ValueError('Incomplete design rows')
    matrix = np.asarray(breadth['counts_by_condition'], dtype=float) / breadth['samples_per_condition']
    values = {d['name']: rule_log_breadth(matrix, [d['weights'][name] for name in names], rules, predicates, mode=mode)
              for d in designs}
    rows = []
    for row in diag['rows']:
        value = values[row['design']][row['rule']]
        if value is None:
            raise ValueError('Undefined breadth')
        rows.append({**row, 'log_breadth': value})
    module_spec = importlib.util.spec_from_file_location('breadth_exploration', Path(__file__).with_name('explore-breadth-design.py'))
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    weights = frozen['allocations']['wider_five_feature']
    row_weights = np.asarray([weights[row['rule']] for row in rows], dtype=float)
    stats = module.full_statistics(rows, row_weights)
    reference_vifs = reference_adjusted_vifs(rows, row_weights, module.FEATURES)
    by_name = {design['name']: design for design in designs}
    references = {prefix: by_name[prefix + '_reference']['weights'] for prefix in ('uniform', 'fitted')}
    sub = subspace(report, references['uniform'])
    singleton = np.array([event['column_probabilities'] for event in sub['singleton_events']])
    deviations = {}
    for prefix, reference in references.items():
        baseline = np.array([reference[name] for name in names])
        maxima = []
        for design in designs:
            if not design['name'].startswith(prefix + '_'):
                continue
            vector = np.array([design['weights'][name] for name in names])
            maxima.append(float(np.max(np.abs(singleton @ (vector - baseline)))))
        if not maxima:
            raise ValueError('Missing reference group')
        deviations[prefix] = {'max_absolute': max(maxima), 'median_max_absolute': float(np.median(maxima))}
    output = {'schema_version': 1, 'input_sha256': [hashlib.sha256(x).hexdigest() for x in raw],
              'seed': report['seed'], 'samples_per_condition': report['samples_per_condition'],
              'breadth_mode': mode, 'rows': len(rows), 'statistics': stats,
              'reference_adjusted_vif': reference_vifs, 'singleton_deviation_from_own_reference': deviations,
              'note': ('Frozen exploratory design on a fresh Monte Carlo draw; not an external benchmark, '
                       'not a registered estimand, and not a Stage 0 gate.')}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
