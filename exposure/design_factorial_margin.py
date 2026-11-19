"""Exploratory max-min margin fit for four shared exposure mixtures.

Local numerical optimization is not a global feasibility or Stage 0 proof.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from exposure.design_factorial import contrasts, features, load_matrix


def optimize(initial, matrices, *, delta_m=.02, delta_l=.02, cross_tolerance=.05,
             min_weight=.001, max_weight=.1, maxiter=300):
    """Maximize the worst constraint margin over the supplied fitting matrices."""
    w = np.asarray(initial, dtype=float)
    if w.ndim != 2 or w.shape[0] != 4 or not matrices:
        raise ValueError('expected four weights and at least one matrix')
    n = w.shape[1]
    settings = np.asarray([delta_m, delta_l, cross_tolerance, min_weight, max_weight], dtype=float)
    if (n < 2 or not np.isfinite(w).all() or not np.isfinite(settings).all()
            or min(delta_m, delta_l, cross_tolerance, min_weight) < 0
            or max_weight <= min_weight or type(maxiter) is not int or maxiter < 1
            or np.any(w < min_weight - 1e-9) or np.any(w > max_weight + 1e-9)
            or not np.allclose(w.sum(axis=1), 1, rtol=0, atol=1e-8)):
        raise ValueError('invalid initial weights or settings')
    for marginal, joint in matrices:
        m, j = np.asarray(marginal), np.asarray(joint)
        if (j.ndim != 2 or j.shape[1] != n or m.shape != (j.shape[0], 2, n)
                or j.shape[0] < 1 or not np.isfinite(m).all() or not np.isfinite(j).all()
                or np.any(m < 0) or np.any(m > 1) or np.any(j < 0) or np.any(j > 1)):
            raise ValueError('invalid exposure matrix')

    def constraints(x):
        return np.concatenate([contrasts(x[:-1], m, j, delta_m, delta_l, cross_tolerance)[0] - x[-1]
                               for m, j in matrices])

    def constraint_jacobian(x):
        result = []
        for m, j in matrices:
            gradient = contrasts(x[:-1], m, j, delta_m, delta_l, cross_tolerance)[1]
            result.append(np.column_stack((gradient, -np.ones(len(gradient)))))
        return np.vstack(result)

    initial_flat = np.r_[w.ravel(), 0.0]
    initial_flat[-1] = float(np.min(constraints(initial_flat)))
    equality_jac = np.column_stack((np.kron(np.eye(4), np.ones((1, n))), np.zeros(4)))
    result = minimize(lambda x: -x[-1], initial_flat,
                      jac=lambda x: np.r_[np.zeros(4 * n), -1.0], method='SLSQP',
                      bounds=[(min_weight, max_weight)] * (4 * n) + [(-1., 1.)],
                      constraints=[{'type': 'eq', 'fun': lambda x: x[:-1].reshape(4, n).sum(axis=1) - 1,
                                    'jac': lambda x: equality_jac},
                                   {'type': 'ineq', 'fun': constraints, 'jac': constraint_jacobian}],
                      options={'maxiter': maxiter, 'ftol': 1e-9})
    fitted = result.x[:-1].reshape(4, n)
    violation = max(float(np.max(np.abs(fitted.sum(axis=1) - 1))),
                    max(0., float(min_weight - fitted.min())), max(0., float(fitted.max() - max_weight)),
                    max(0., -float(constraints(result.x).min())))
    eigenvalues = []
    for m, j in matrices:
        x = features(fitted, m, j)[0]
        centered = x - x.mean(axis=0, keepdims=True)
        eigenvalues.append(np.linalg.eigvalsh(np.einsum('kri,krj->ij', centered, centered)).tolist())
    return {'feasible': bool(result.success and violation <= 1e-7 and result.x[-1] >= -1e-7
                             and all(e[0] > 1e-10 for e in eigenvalues)),
            'solver_success': bool(result.success), 'weights': fitted.tolist(),
            'margin': float(result.x[-1]), 'max_violation': violation,
            'information_eigenvalues': eigenvalues, 'message': str(result.message)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-design', type=Path, required=True)
    parser.add_argument('--fit-report', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        source = args.initial_design.read_bytes()
        base = json.loads(source)
        raw = [path.read_bytes() for path in args.fit_report]
        reports = [json.loads(item) for item in raw]
        reference = reports[0]
        seeds = [report.get('seed') for report in reports]
        if (not reference.get('estimator_sha256') or not reference.get('estimator')
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError('fitting reports require an estimator fingerprint and distinct seeds')
        order, *first = load_matrix(reference, base['rule_names'])
        matrices = [tuple(first)]
        for report in reports[1:]:
            candidate_order, *matrix = load_matrix(report, base['rule_names'])
            if (candidate_order != order or report.get('rules') != reference.get('rules')
                    or report.get('estimator_sha256') != reference.get('estimator_sha256')
                    or report.get('estimator') != reference.get('estimator')):
                raise ValueError('fitting report definitions, order or estimator differ')
            matrices.append(tuple(matrix))
        if order != base['condition_order'] or not base.get('best', {}).get('feasible'):
            raise ValueError('initial design does not match fitting conditions')
        fit = optimize(base['best']['weights'], matrices)
        result = copy.deepcopy(base)
        # The seed design's optimizer starts and candidates are not this fit's.
        result.pop('candidates', None)
        result.update({'best': fit, 'report_sha256': hashlib.sha256(raw[0]).hexdigest(),
                       'fitting_report_sha256': [hashlib.sha256(item).hexdigest() for item in raw],
                       'fitting_seeds': seeds,
                       'initial_design_sha256': hashlib.sha256(source).hexdigest(),
                       'method_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'method': 'local_max_min_constraint_margin',
                       'note': 'Exploratory local fit; independent holdout and full research gates remain required.'})
        result['settings'].update({'delta_m': .02, 'delta_l': .02, 'cross_tolerance': .05,
                                   'min_weight': .001, 'max_weight': .1, 'starts': 1,
                                   'seed': None, 'maxiter': 300, 'fitting_matrices': len(matrices)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write('\n')
        print(json.dumps({key: fit[key] for key in ('feasible', 'margin', 'max_violation', 'message')}))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
