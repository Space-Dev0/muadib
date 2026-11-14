"""Explore bounded nonuniform rule allocation for endpoint exposure designs.

This changes the analysis population. It is not a Stage 0 gate assessment.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from exposure.design_diagnostics import FEATURES, diagnose
from exposure.design_extremes import generate
from exposure.design_reference_explore import matrix, optimize_reference
from exposure.measure import unique_object


def moments(rows, names):
    """Per-rule mean and second moment for complete equal-design rows."""
    groups = {name: [] for name in names}
    seen = set()
    for row in rows:
        rule, design = row['rule'], row['design']
        if rule not in groups or (rule, design) in seen:
            raise ValueError('Unexpected or duplicate rule-by-design row')
        seen.add((rule, design))
        groups[rule].append([row[f] for f in FEATURES])
    sets = [{design for rule, design in seen if rule == name} for name in names]
    if not sets or not sets[0] or any(s != sets[0] for s in sets):
        raise ValueError('Expected complete matching design rows for each rule')
    arrays = [np.asarray(groups[name], dtype=float) for name in names]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite design covariates')
    return np.stack([a.mean(axis=0) for a in arrays]), np.stack([a.T @ a / len(a) for a in arrays])


def statistics(means, seconds, weights):
    """Weighted pooled correlation and VIFs including an intercept."""
    w = np.asarray(weights, dtype=float)
    if (w.ndim != 1 or len(w) != len(means) or np.any(w < 0)
            or not np.all(np.isfinite(w)) or abs(w.sum()-1) > 1e-6):
        raise ValueError('Invalid rule allocation')
    w = w / w.sum()
    mu = w @ means
    cov = np.einsum('i,ijk->jk', w, seconds) - np.outer(mu, mu)
    var = np.diag(cov)
    if np.any(var <= 1e-14):
        raise ValueError('Constant or numerically unstable feature')
    corr = cov / np.sqrt(np.outer(var, var))
    if np.linalg.eigvalsh(corr).min() <= 1e-10:
        raise ValueError('Collinear or numerically unstable features')
    vif = np.diag(np.linalg.inv(corr))
    return {'correlation_marginal_lift': float(corr[0, 1]),
            'vif': dict(zip(FEATURES, map(float, vif), strict=True))}


def fit(means, seconds, *, min_multiple=0.2, max_multiple=2.0, target_vif=1.75,
        target_correlation=0.25):
    n = len(means)
    if not (n >= 2 and 0 < min_multiple <= 1 <= max_multiple and target_vif > 1
            and 0 <= target_correlation < 1):
        raise ValueError('Invalid allocation bounds or targets')
    def objective(q):
        try:
            result = statistics(means, seconds, q)
        except (ValueError, np.linalg.LinAlgError):
            return 1e6
        v = np.array(list(result['vif'].values()))
        return float(np.sum(np.maximum(v-target_vif, 0)**2)
                     + max(abs(result['correlation_marginal_lift'])-target_correlation, 0)**2)
    start = np.full(n, 1/n)
    result = minimize(objective, start, method='SLSQP', bounds=[(min_multiple/n,max_multiple/n)]*n,
                      constraints={'type':'eq','fun':lambda q: np.sum(q)-1},
                      options={'maxiter':1000,'ftol':1e-11})
    q = result.x
    if (not result.success or abs(q.sum()-1)>1e-8 or np.min(q)<min_multiple/n-1e-8
            or np.max(q)>max_multiple/n+1e-8):
        raise ValueError(f'Allocation optimizer failed: {result.message}')
    return q, float(result.fun)


def integer_allocation(weights, total):
    """Largest-remainder allocation with stable tie-breaking, not a power analysis."""
    w = np.asarray(weights, dtype=float)
    if (type(total) is not int or total < 1 or w.ndim != 1 or not len(w)
            or not np.all(np.isfinite(w)) or np.any(w < 0) or abs(w.sum()-1) > 1e-8):
        raise ValueError('Invalid weights or total')
    quota = w * total
    counts = np.floor(quota).astype(int)
    remaining = total - int(counts.sum())
    counts[np.argsort(-(quota-counts), kind='stable')[:remaining]] += 1
    return counts


def explore(training, holdout):
    names, values, counts = matrix(training)
    hn, _, hc = matrix(holdout)
    if names != hn or not np.array_equal(counts, hc) or training['rules'] != holdout['rules']:
        raise ValueError('Reports must use the same ordered rules')
    reference = dict(zip(names, map(float, optimize_reference(values, counts)), strict=True))
    designs = generate(training, reference, min_weight=0.0001)
    results = [diagnose(report, designs) for report in (training, holdout)]
    if any(r['excluded_count'] for r in results):
        raise ValueError('Cannot analyze incomplete rule-by-design rows')
    moments_by_report = [moments(r['rows'], names) for r in results]
    weights, objective = fit(*moments_by_report[0])
    uniform = np.full(len(names), 1/len(names))
    output = {'schema_version':1, 'reference_weights':reference, 'designs': designs,
              'rule_allocation':dict(zip(names, map(float,weights), strict=True)),
              'fit_objective':objective, 'allocation_bounds_relative_to_equal':[0.2,2.0],
              'training':{},'holdout':{}, 'note':('Exploratory rule allocation changes the target population; '
              'four features only, no breadth, dependent rows, singleton estimates and dose limitations. '
              'Does not establish the Stage 0 gate or identify a marginal effect with rule fixed effects.')}
    for label, (a,b) in zip(('training','holdout'), moments_by_report, strict=True):
        output[label] = {'equal_rules':statistics(a,b,uniform), 'fitted_rules':statistics(a,b,weights)}
    counts = integer_allocation(weights, 285)
    output['integer_example'] = {'samples_per_design': 285,
        'counts_by_rule': dict(zip(names, map(int, counts), strict=True)),
        'holdout_statistics': statistics(*moments_by_report[1], counts / counts.sum()),
        'note': 'Illustrative per-design rule allocation only; not sample-size or power advice.'}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-report',type=Path,required=True)
    p.add_argument('--holdout-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        raw=[p.read_bytes() for p in (a.training_report,a.holdout_report)]
        reports=[json.loads(b,object_pairs_hook=unique_object) for b in raw]
        out=explore(*reports)
        out['input_sha256']=[hashlib.sha256(b).hexdigest() for b in raw]
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f: json.dump(out,f,indent=2,allow_nan=False); f.write('\n')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, np.linalg.LinAlgError) as exc:
        p.error(str(exc))
    print(f'Wrote {a.output}')


if __name__ == '__main__':
    main()
