"""Exploratory D-optimal allocation over *given* candidate feature rows.

This does not create candidate designs or resolve nonidentifiability in them.
The returned allocation changes the target population when applied to rule rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize


def solve(features, *, constraints=(), min_weight=0.0):
    """Maximize log det(X.T diag(w) X), subject to simplex and linear bounds.

    Every row of X is a selectable candidate design and each column a fixed
    predictor (include an intercept explicitly). No covariance assumption or
    estimated uncertainty is implied by the resulting information matrix.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError('features must be a finite nonempty matrix')
    n, p = x.shape
    if n < p or np.linalg.matrix_rank(x) < p:
        raise ValueError('feature matrix is rank deficient; D-optimality cannot identify its columns')
    if not np.isfinite(min_weight) or not 0 <= min_weight <= 1 / n:
        raise ValueError('invalid minimum weight')
    aub, bub, parsed = [], [], []
    for constraint in constraints:
        if not isinstance(constraint, dict) or set(constraint) - {'coefficients', 'min', 'max'}:
            raise ValueError('invalid constraint')
        row = np.asarray(constraint.get('coefficients'), dtype=float)
        if row.shape != (n,) or not np.isfinite(row).all():
            raise ValueError('constraint coefficients must be finite and match the candidate count')
        low, high = constraint.get('min'), constraint.get('max')
        if low is None and high is None:
            raise ValueError('constraint must have a bound')
        if any(v is not None and not np.isfinite(v) for v in (low, high)):
            raise ValueError('constraint bounds must be finite')
        if low is not None and high is not None and low > high:
            raise ValueError('constraint bounds are reversed')
        if low is not None:
            aub.append(-row); bub.append(-float(low))
        if high is not None:
            aub.append(row); bub.append(float(high))
        parsed.append((row, low, high))
    # Find a feasible starting point maximizing its smallest weight. Unlike an
    # arbitrary LP vertex this often yields a nonsingular information matrix.
    aug_ub = [np.r_[row, 0.] for row in aub]
    aug_ub += [np.r_[-np.eye(n)[i], 1.] for i in range(n)]
    start = linprog(np.r_[np.zeros(n), -1.], A_ub=np.asarray(aug_ub),
                    b_ub=np.r_[bub, np.zeros(n)], A_eq=[np.r_[np.ones(n), 0.]],
                    b_eq=[1.], bounds=[(min_weight, None)] * n + [(0., None)], method='highs')
    if not start.success:
        raise ValueError('linear constraints are infeasible')
    initial = start.x[:n]
    def info(w):
        return x.T @ (w[:, None] * x)
    if np.linalg.matrix_rank(info(initial)) < p:
        raise ValueError('feasible starting information matrix is singular; full-rank design not established')
    def objective(w):
        sign, value = np.linalg.slogdet(info(w))
        return -value if sign > 0 else 1e100
    def gradient(w):
        matrix = info(w)
        try:
            return -np.einsum('ij,ji->i', x, np.linalg.solve(matrix, x.T))
        except np.linalg.LinAlgError:
            return np.zeros(n)
    slsqp_constraints = [{'type':'eq', 'fun':lambda w: w.sum()-1., 'jac':lambda w: np.ones(n)}]
    for row, low, high in parsed:
        if low is not None and high is not None and low == high:
            slsqp_constraints.append({'type':'eq', 'fun':lambda w, r=row, v=low: r@w-v,
                                      'jac':lambda w, r=row: r})
        else:
            if low is not None:
                slsqp_constraints.append({'type':'ineq', 'fun':lambda w, r=row, v=low: r@w-v,
                                          'jac':lambda w, r=row: r})
            if high is not None:
                slsqp_constraints.append({'type':'ineq', 'fun':lambda w, r=row, v=high: v-r@w,
                                          'jac':lambda w, r=row: -r})
    result = minimize(objective, initial, jac=gradient, method='SLSQP',
                      bounds=[(min_weight, 1.)] * n, constraints=slsqp_constraints,
                      options={'ftol':1e-10, 'maxiter':2000})
    if not result.success:
        raise RuntimeError(f'D-optimal solver did not converge: {result.message}')
    w = result.x
    violations = [abs(w.sum()-1), max(0., min_weight-float(w.min()))]
    for row, low, high in parsed:
        val = float(row @ w)
        if low is not None: violations.append(max(0., low-val))
        if high is not None: violations.append(max(0., val-high))
    sign, logdet = np.linalg.slogdet(info(w))
    if max(violations) > 1e-7 or sign <= 0 or not np.isfinite(logdet):
        raise RuntimeError('solver returned an infeasible or singular design')
    return {'weights': w.tolist(), 'log_determinant': float(logdet),
            'information_eigenvalues': np.linalg.eigvalsh(info(w)).tolist(),
            'max_constraint_violation': max(violations), 'iterations': int(result.nit),
            'note': ('Exploratory D-optimal allocation over supplied rows; information matrix '
                     'depends on feature definitions and weighting. It does not establish a valid '
                     'exposure design, uncertainty, model identifiability, or Stage 0 readiness.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSON with features and optional constraints/min_weight')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    if set(data) - {'features', 'constraints', 'min_weight'} or 'features' not in data:
        parser.error('input requires features and optional constraints and min_weight')
    try:
        result = solve(data['features'], constraints=data.get('constraints', ()), min_weight=data.get('min_weight', 0.))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    main()
