"""Diagnose within-rule/reference exposure variation; exploratory only."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exposure.design_diagnostics import diagnose
from exposure.design_extremes import generate
from exposure.design_reference_explore import matrix, optimize_reference


def main():
    pilot = json.loads(Path('outputs/exposure-57-pilot.json').read_text())
    holdout = json.loads(Path('outputs/exposure-57-holdout-81205.json').read_text())
    names, values, counts = matrix(pilot)
    references = {
        'uniform': dict.fromkeys(names, 1 / len(names)),
        'fitted': dict(zip(names, map(float, optimize_reference(values, counts)), strict=True)),
    }
    designs = []
    for prefix, weights in references.items():
        for design in generate(pilot, weights, min_weight=0.0001)['designs']:
            designs.append({'name': prefix + '_' + design['name'], 'weights': design['weights']})

    for label, report in [('pilot', pilot), ('holdout', holdout)]:
        result = diagnose(report, {'designs': designs})
        if result['excluded_count']:
            raise ValueError('Incomplete rows make this comparison invalid')
        groups = defaultdict(list)
        for row in result['rows']:
            groups[(row['rule'], row['design'].split('_')[0])].append(row)
        marginal, lift, ranges = [], [], []
        for rows in groups.values():
            a = np.array([row['log_mean_marginal'] for row in rows])
            b = np.array([row['log_lift'] for row in rows])
            marginal.extend(a - a.mean())
            lift.extend(b - b.mean())
            ranges.append(np.ptp(a))
        marginal, lift = np.asarray(marginal), np.asarray(lift)
        print(label, 'groups', len(groups), 'rows', len(marginal),
              'marginal_rms', float(np.sqrt(np.mean(marginal**2))),
              'max_marginal_range', float(max(ranges)),
              'lift_rms', float(np.sqrt(np.mean(lift**2))),
              'rank_at_1e-10', int(np.linalg.matrix_rank(np.column_stack((marginal, lift)), tol=1e-10)))

        # Remove a separate intercept per rule and a single shared reference
        # intercept. This tests a different, less flexible adjustment model.
        rows = result['rows']
        rules = sorted({row['rule'] for row in rows})
        controls = np.array([[float(row['rule'] == name) for name in rules] +
                             [float(row['design'].startswith('fitted_'))] for row in rows])
        features = np.array([[row['log_mean_marginal'], row['log_lift']] for row in rows])
        residual = features - controls @ np.linalg.lstsq(controls, features, rcond=None)[0]
        corr = float(np.corrcoef(residual.T)[0, 1])
        print(label, 'rule_and_shared_reference_adjusted_rank',
              int(np.linalg.matrix_rank(residual, tol=1e-10)),
              'residual_correlation', corr, 'two_feature_vif', 1 / (1 - corr**2),
              'marginal_rms', float(np.sqrt(np.mean(residual[:, 0]**2))))


if __name__ == '__main__':
    main()
