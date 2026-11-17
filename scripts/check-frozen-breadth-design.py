"""Evaluate a frozen exploratory allocation on a fresh Monte Carlo matrix."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from exposure.design_breadth import key, rule_log_breadth
from exposure.design_diagnostics import diagnose
from exposure.measure import load_rules


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--breadth', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = [path.read_bytes() for path in (args.frozen, args.report, args.breadth)]
    frozen, report, breadth = [json.loads(data) for data in raw]
    if report['seed'] in (43, 81205) or breadth['source_sha256'] != hashlib.sha256(raw[1]).hexdigest():
        raise ValueError('Expected a fresh measurement with matching breadth source')
    names = frozen['rule_order']; rules = load_rules(report['rules'])
    if list(rules) != names or breadth['rule_order'] != names:
        raise ValueError('Rule order mismatch')
    predicates = sorted({predicate for atoms in rules.values() for predicate in atoms}, key=key)
    if breadth['predicate_keys'] != [key(predicate) for predicate in predicates]:
        raise ValueError('Predicate vocabulary mismatch')
    if breadth['samples_per_condition'] != report['samples_per_condition'] or breadth['source_seed'] != report['seed']:
        raise ValueError('Breadth sampling mismatch')
    designs = frozen['designs']; diag = diagnose(report, {'designs': designs})
    if diag['excluded_count']:
        raise ValueError('Incomplete design rows')
    matrix = np.asarray(breadth['counts_by_condition'], dtype=float) / breadth['samples_per_condition']
    values = {d['name']: rule_log_breadth(matrix, [d['weights'][name] for name in names], rules, predicates)
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
    stats = module.full_statistics(rows, [weights[row['rule']] for row in rows])
    output = {'schema_version': 1, 'input_sha256': [hashlib.sha256(x).hexdigest() for x in raw],
              'seed': report['seed'], 'samples_per_condition': report['samples_per_condition'],
              'rows': len(rows), 'statistics': stats,
              'note': ('Frozen exploratory design on a fresh Monte Carlo draw; not an external benchmark, '
                       'not a registered estimand, and not a Stage 0 gate.')}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
