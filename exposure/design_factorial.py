"""Exploratory four-global-mixture design; no claim of gate validity.

The two covariates are within-rule mean log singleton exposure and log lift.
The optimizer fixes equal rule weight and uses exact analytic gradients. It
only examines the supplied empirical matrix and does not propagate uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def load_matrix(report, rule_names):
    """Return singleton [rules, 2, conditions] and joint [rules, conditions]."""
    order = report['rule_order']
    if len(set(order)) != len(order) or not rule_names or len(set(rule_names)) != len(rule_names):
        raise ValueError('rule and condition names must be nonempty and unique')
    if any(name not in report['rules'] or len(report['rules'][name]) != 2 for name in rule_names):
        raise ValueError('all target rules must have exactly two predicates')
    columns = {c['condition']: c for c in report['columns']}
    if set(columns) != set(order) or len(columns) != len(report['columns']):
        raise ValueError('condition columns are missing or duplicated')
    marg = np.empty((len(rule_names), 2, len(order)))
    joint = np.empty((len(rule_names), len(order)))
    for j, name in enumerate(order):
        col = columns[name]
        for r, rule in enumerate(rule_names):
            joint[r, j] = col['measurements'][rule]['probability']
            subsets = {tuple(p['predicate_indices']): p['probability'] for p in col['predicate_subsets'][rule]}
            for p in range(2): marg[r, p, j] = subsets[(p,)]
    if not np.isfinite(marg).all() or not np.isfinite(joint).all() or (marg < 0).any() or (joint < 0).any() or (marg > 1).any() or (joint > 1).any():
        raise ValueError('probabilities must be finite and in [0,1]')
    return order, marg, joint


def features(weights, marg, joint, eps=1e-10):
    """Return features [4,R,2] and derivative [4,R,2,J]."""
    m = np.einsum('rpj,kj->krp', marg, weights)
    n = np.einsum('rj,kj->kr', joint, weights)
    if np.min(m) <= eps or np.min(n) <= eps:
        raise ValueError('zero/near-zero exposure makes logarithms undefined')
    mean_log = np.log(m).mean(axis=2)
    lift = np.log(n) - np.log(m).sum(axis=2)
    dm = (marg[None] / m[..., None]).mean(axis=2)
    dl = joint[None] / n[..., None] - (marg[None] / m[..., None]).sum(axis=2)
    return np.stack([mean_log, lift], axis=2), np.stack([dm, dl], axis=2)


def score_gradient(flat, marg, joint, ridge=1e-8):
    w = np.asarray(flat).reshape(4, joint.shape[1])
    x, dx = features(w, marg, joint)
    z = x - x.mean(axis=0, keepdims=True)
    info = np.einsum('kri,krj->ij', z, z) + ridge * np.eye(2)
    sign, value = np.linalg.slogdet(info)
    if sign <= 0: raise ValueError('information matrix is not positive definite')
    dscore = 2 * np.einsum('kri,ij,krjp->kp', z, np.linalg.inv(info), dx)
    return -float(value), -dscore.ravel()


def contrasts(flat, marg, joint, delta_m, delta_l, cross_tolerance=None):
    """Per-rule factorial contrasts for LL,LH,HL,HH; nonnegative means feasible."""
    w = np.asarray(flat).reshape(4, joint.shape[1]); x, dx = features(w, marg, joint)
    pairs = [(2, 0, 0, delta_m), (3, 1, 0, delta_m),
             (1, 0, 1, delta_l), (3, 2, 1, delta_l)]
    values, derivatives = [], []
    for high, low, feature, delta in pairs:
        for r in range(joint.shape[0]):
            values.append(x[high, r, feature] - x[low, r, feature] - delta)
            d = np.zeros_like(w); d[high] = dx[high, r, feature]; d[low] = -dx[low, r, feature]
            derivatives.append(d.ravel())
    if cross_tolerance is not None:
        # Cross-factor differences should remain small on each paired contrast.
        for high, low, feature in [(1, 0, 0), (3, 2, 0), (2, 0, 1), (3, 1, 1)]:
            for r in range(joint.shape[0]):
                difference = x[high, r, feature] - x[low, r, feature]
                d = np.zeros_like(w); d[high] = dx[high, r, feature]; d[low] = -dx[low, r, feature]
                values.extend([cross_tolerance - difference, cross_tolerance + difference])
                derivatives.extend([-d.ravel(), d.ravel()])
    return np.asarray(values), np.asarray(derivatives)


def optimize(marg, joint, *, delta_m=.02, delta_l=.02, min_weight=.001,
             max_weight=1., starts=4, seed=42, maxiter=200, cross_tolerance=.05,
             extra_matrices=()):
    """Optimize shared weights against one or more empirical matrices.

    Multiple fitting draws are not independent validation. Success indicates
    only local numerical feasibility for the supplied constraints and matrices.
    """
    matrices = [(np.asarray(marg, float), np.asarray(joint, float))]
    matrices += [(np.asarray(m, float), np.asarray(j, float)) for m, j in extra_matrices]
    first_m, first_j = matrices[0]
    if first_j.ndim != 2:
        raise ValueError('incompatible matrices')
    r, n = first_j.shape
    for m, j in matrices:
        if j.shape != (r, n) or m.shape != (r, 2, n):
            raise ValueError('incompatible matrices')
        if (not np.isfinite(m).all() or not np.isfinite(j).all() or
                (m < 0).any() or (j < 0).any() or (m > 1).any() or (j > 1).any()):
            raise ValueError('probabilities must be finite and in [0,1]')
    if (r < 1 or n < 2 or not np.isfinite(min_weight) or not np.isfinite(max_weight)
            or min_weight < 0 or min_weight * n >= 1 or max_weight <= min_weight
            or max_weight * n < 1 or starts < 1 or not np.isfinite([delta_m, delta_l, cross_tolerance]).all()
            or min(delta_m, delta_l, cross_tolerance) < 0):
        raise ValueError('invalid optimization settings')
    rng = np.random.default_rng(seed)
    eq_jac = np.kron(np.eye(4), np.ones((1, n)))
    eq = {'type': 'eq', 'fun': lambda x: x.reshape(4, n).sum(axis=1) - 1, 'jac': lambda x: eq_jac}
    def all_contrasts(x):
        values = [contrasts(x, m, j, delta_m, delta_l, cross_tolerance) for m, j in matrices]
        return np.concatenate([v[0] for v in values]), np.concatenate([v[1] for v in values], axis=0)
    def objective(x):
        scores = [score_gradient(x, m, j) for m, j in matrices]
        return np.mean([v[0] for v in scores]), np.mean([v[1] for v in scores], axis=0)
    ineq = {'type': 'ineq', 'fun': lambda x: all_contrasts(x)[0],
            'jac': lambda x: all_contrasts(x)[1]}
    candidates = []
    for i in range(starts):
        # Blend with uniform so every starting point obeys the upper bound.
        draw = rng.dirichlet(np.ones(n), size=4)
        spread = np.max(draw - 1/n)
        scale = min(0.8, (max_weight - 1/n) / spread) if spread > 0 else 0
        initial = (1-scale) / n + scale * draw
        if np.min(initial) < min_weight:
            initial = np.full((4, n), 1/n)
        try:
            result = minimize(objective, initial.ravel(), jac=True, method='SLSQP',
                              bounds=[(min_weight, max_weight)] * (4*n), constraints=[eq, ineq],
                              options={'maxiter': maxiter, 'ftol': 1e-9})
            w = result.x.reshape(4, n)
            eigenvalues = []
            for fit_m, fit_j in matrices:
                x, _ = features(w, fit_m, fit_j)
                z = x - x.mean(axis=0, keepdims=True)
                eigenvalues.append(np.linalg.eigvalsh(np.einsum('kri,krj->ij', z, z)))
            eig = eigenvalues[0]
            cv = all_contrasts(result.x)[0]
            err = max(float(np.abs(w.sum(axis=1)-1).max()), max(0., float(min_weight-w.min())),
                      max(0., float(w.max()-max_weight)), max(0., -float(cv.min())))
            candidates.append({'start':i, 'solver_success':bool(result.success), 'message':str(result.message),
                               'iterations':int(result.nit), 'max_violation':err, 'min_factorial_margin':float(cv.min()),
                               'information_eigenvalues':eig.tolist(), 'information_eigenvalues_by_matrix':[e.tolist() for e in eigenvalues], 'mean_logdet_objective':float(-result.fun),
                               'weights':w.tolist(),
                               'feasible':bool(result.success and err <= 1e-7 and all(e[0] > 1e-10 for e in eigenvalues))})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            candidates.append({'start':i,'solver_success':False,'feasible':False,'message':str(exc)})
    valid = [c for c in candidates if c['feasible']]
    best = max(valid, key=lambda c: c['mean_logdet_objective']) if valid else None
    return {'candidates':candidates, 'best':best, 'settings':{'delta_m':delta_m,'delta_l':delta_l,
            'min_weight':min_weight,'max_weight':max_weight,'starts':starts,'seed':seed,'maxiter':maxiter,
            'cross_tolerance':cross_tolerance, 'fitting_matrices':len(matrices)},
            'note':'Exploratory empirical fit. No held-out validation, uncertainty, causal identification or Stage 0 pass.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--rules',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--starts',type=int,default=4);p.add_argument('--maxiter',type=int,default=200)
    p.add_argument('--delta-m',type=float,default=.02);p.add_argument('--delta-l',type=float,default=.02)
    p.add_argument('--cross-tolerance',type=float,default=.05)
    p.add_argument('--max-weight',type=float,default=1.)
    p.add_argument('--fit-reports',nargs='*',type=Path,default=[])
    args=p.parse_args();raw=args.report.read_bytes(); report=json.loads(raw); order,marg,joint=load_matrix(report,args.rules)
    extras=[]; hashes=[hashlib.sha256(raw).hexdigest()]; seeds=[report.get('seed')]
    for path in args.fit_reports:
        extra_raw=path.read_bytes(); extra_report=json.loads(extra_raw)
        extra_order,extra_marg,extra_joint=load_matrix(extra_report,args.rules)
        if extra_order != order or extra_report['rules'] != report['rules']:
            raise ValueError('fitting report rules or condition order differ')
        if extra_report.get('seed') in seeds:
            raise ValueError('fitting report seeds must be distinct')
        extras.append((extra_marg,extra_joint)); hashes.append(hashlib.sha256(extra_raw).hexdigest()); seeds.append(extra_report.get('seed'))
    result=optimize(marg,joint,starts=args.starts,maxiter=args.maxiter,delta_m=args.delta_m,delta_l=args.delta_l,cross_tolerance=args.cross_tolerance,max_weight=args.max_weight,extra_matrices=extras)
    result.update({'report_sha256':hashlib.sha256(raw).hexdigest(),'fitting_report_sha256':hashes,'fitting_seeds':seeds,'rule_names':args.rules,'condition_order':order})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out: json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({'feasible':result['best'] is not None,'starts':[{'start':c['start'],'feasible':c['feasible'],'message':c['message'],'max_violation':c.get('max_violation')} for c in result['candidates']]}))

if __name__ == '__main__': main()
