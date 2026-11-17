"""Construct exploratory endpoints preserving singleton marginals on several matrices."""
import math
import numpy as np
from scipy.optimize import linprog

from exposure.design_subspace import diagnose


def generate(reports, reference, *, min_weight=0.0001):
    """Constrain each estimated matrix against the same reference mixture.

    This is sample-fitting, not a guarantee of population preservation.
    """
    if not isinstance(reports, list) or len(reports) < 2:
        raise ValueError('At least two measurement matrices are required')
    if not math.isfinite(min_weight) or min_weight < 0:
        raise ValueError('Invalid minimum weight')
    if any(r['rules'] != reports[0]['rules'] or r['rule_order'] != reports[0]['rule_order'] for r in reports):
        raise ValueError('Measurement rules or order differ')
    if len({r['seed'] for r in reports}) != len(reports):
        raise ValueError('Measurement seeds must differ')
    subspaces = [diagnose(report, reference) for report in reports]
    names = subspaces[0]['column_names']
    expected_events = [x['event'] for x in subspaces[0]['singleton_events']]
    if any(s['column_names'] != names or [x['event'] for x in s['singleton_events']] != expected_events
           for s in subspaces):
        raise ValueError('Singleton event definitions differ')
    base = np.array([subspaces[0]['reference_weights'][name] for name in names])
    if np.min(base) < min_weight - 1e-12:
        raise ValueError('Reference violates minimum-weight floor')
    rows = [np.ones(len(names))]
    for sub in subspaces:
        rows.extend(np.asarray(event['column_probabilities']) for event in sub['singleton_events'])
    matrix = np.asarray(rows); rhs = matrix @ base
    columns = {column['condition']: column for column in reports[0]['columns']}
    designs = [{'name': 'reference', 'weights': dict(zip(names, map(float, base), strict=True))}]
    for index, rule in enumerate(names):
        objective = np.array([columns[name]['measurements'][rule]['probability'] for name in names])
        for label, sign in (('min', 1), ('max', -1)):
            solution = linprog(sign * objective, A_eq=matrix, b_eq=rhs,
                               bounds=[(min_weight, None)] * len(names), method='highs')
            if not solution.success:
                raise RuntimeError(f'LP failed for {rule}: {solution.message}')
            values = solution.x
            if not np.isfinite(values).all() or np.max(np.abs(matrix @ values - rhs)) > 1e-7:
                raise RuntimeError('LP solution violates equality constraints')
            designs.append({'name': f'endpoint_{index:03d}_{label}',
                            'weights': dict(zip(names, map(float, values), strict=True))})
    return {'designs': designs}
