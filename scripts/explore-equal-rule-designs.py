"""Explore design-level sampling weights while keeping rule representation equal."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from exposure.design_breadth import key, predicate_vocabulary, rule_log_breadth
from exposure.design_diagnostics import FEATURES, diagnose
from exposure.measure import load_rules

FEATURES6 = (*FEATURES, 'log_breadth', 'reference_group')


def group_moments(rows, design_names, rule_names):
    groups = {name: {} for name in design_names}
    for row in rows:
        group = groups[row['design']]
        if row['rule'] in group:
            raise ValueError('Duplicate design/rule row')
        group[row['rule']] = [row[f] for f in FEATURES6]
    if any(set(group) != set(rule_names) for group in groups.values()):
        raise ValueError('Incomplete design/rule rows')
    arrays = [np.asarray([groups[d][r] for r in rule_names], dtype=float) for d in design_names]
    return (np.stack([a.mean(axis=0) for a in arrays]),
            np.stack([a.T @ a / len(a) for a in arrays]))


def stats_gradient(means, seconds, weights):
    """Weighted correlations, VIFs and analytic derivatives of both."""
    mean = weights @ means
    cov = np.einsum('i,ijk->jk', weights, seconds) - np.outer(mean, mean)
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() <= 1e-12:
        raise ValueError('Singular covariance')
    precision = np.linalg.inv(cov); var = np.diag(cov); vif = var * np.diag(precision)
    derivatives = seconds - means[:, :, None] * mean[None, None, :] - mean[None, :, None] * means[:, None, :]
    dvar = np.diagonal(derivatives, axis1=1, axis2=2)
    dprecision = -np.einsum('ij,njk,kl->nil', precision, derivatives, precision)
    dvif = dvar * np.diag(precision) + var * np.diagonal(dprecision, axis1=1, axis2=2)
    corr = cov[0, 1] / np.sqrt(var[0] * var[1])
    dcorr = derivatives[:, 0, 1] / np.sqrt(var[0] * var[1]) - .5 * corr * (dvar[:, 0] / var[0] + dvar[:, 1] / var[1])
    return float(corr), vif, dcorr, dvif


def objective_gradient(weights, moments):
    value = 0.; gradient = np.zeros_like(weights)
    for means, seconds in moments:
        try:
            corr, vif, dcorr, dvif = stats_gradient(means, seconds, weights)
        except (ValueError, np.linalg.LinAlgError):
            return 1e6, np.zeros_like(weights)
        over = np.maximum(vif - 1.85, 0)
        value += float(over @ over); gradient += 2 * (dvif @ over)
        if abs(corr) > .24:
            delta = abs(corr) - .24
            value += 25 * delta**2; gradient += 50 * delta * np.sign(corr) * dcorr
    return value, gradient


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-design-weight', type=float, default=.05)
    args=p.parse_args()
    if not 0 < args.max_design_weight <= .5:
        raise ValueError('Maximum design weight must lie in (0, 0.5]')
    endpoint_raw = Path('outputs/design-breadth-multimatrix-100k-82570.json').read_bytes()
    endpoints = json.loads(endpoint_raw); designs = endpoints['designs']; design_names=[d['name'] for d in designs]
    report_paths=('outputs/exposure-57-fresh-82511.json','outputs/exposure-57-fresh-82505.json',
                  'outputs/exposure-57-fresh-82512.json')
    breadth_paths=('outputs/breadth-57-fresh-82511.json','outputs/breadth-57-fresh-82505.json',
                   'outputs/breadth-57-fresh-82512.json')
    raw=[Path(path).read_bytes() for path in report_paths]; braw=[Path(path).read_bytes() for path in breadth_paths]
    reports=[json.loads(x) for x in raw]; breadths=[json.loads(x) for x in braw]
    rules=load_rules(reports[0]['rules']); names=list(rules); atoms=predicate_vocabulary(rules,'role')
    all_moments=[]
    for source,report,b in zip(raw,reports,breadths,strict=True):
        if hashlib.sha256(source).hexdigest()!=b['source_sha256'] or b['predicate_keys']!=[key(x) for x in atoms]:
            raise ValueError('Breadth/source mismatch')
        if load_rules(report['rules'])!=rules:raise ValueError('Rules differ')
        diag=diagnose(report,{'designs':designs})
        if diag['excluded_count']:raise ValueError('Incomplete design rows')
        matrix=np.asarray(b['counts_by_condition'],dtype=float)/b['samples_per_condition']
        breadth={d['name']:rule_log_breadth(matrix,[d['weights'][n] for n in names],rules,atoms) for d in designs}
        rows=[{**r,'log_breadth':breadth[r['design']][r['rule']],
               'reference_group':float(r['design'].startswith('fitted_'))} for r in diag['rows']]
        if any(r['log_breadth'] is None for r in rows):raise ValueError('Undefined breadth')
        all_moments.append(group_moments(rows,design_names,names))
    group=np.array([name.startswith('fitted_') for name in design_names],dtype=bool)
    if group.sum()*2 != len(group):raise ValueError('Reference groups must have equal design counts')
    initial=np.full(len(designs),1/len(designs))
    try:
        opt=minimize(lambda w:objective_gradient(w,all_moments[:2]),initial,jac=True,method='SLSQP',
                     bounds=[(0,args.max_design_weight)]*len(designs),
                     constraints=[{'type':'eq','fun':lambda w:w[group].sum()-.5,'jac':lambda w:group.astype(float)},
                                  {'type':'eq','fun':lambda w:w[~group].sum()-.5,'jac':lambda w:(~group).astype(float)}],
                     options={'maxiter':1000,'ftol':1e-10})
    except np.linalg.LinAlgError as exc:raise ValueError('Optimization encountered a singular design') from exc
    if not opt.success or abs(opt.x[group].sum()-.5)>1e-8 or abs(opt.x[~group].sum()-.5)>1e-8:
        raise ValueError(f'Optimization failed: {opt.message}')
    statistics=[]
    for m in all_moments:
        corr,vif,_,_=stats_gradient(*m,opt.x)
        statistics.append({'marginal_lift_r':corr,'vif':dict(zip(FEATURES6,map(float,vif),strict=True))})
    result={'schema_version':1,'endpoint_sha256':hashlib.sha256(endpoint_raw).hexdigest(),
            'report_sha256':[hashlib.sha256(x).hexdigest() for x in raw],
            'breadth_sha256':[hashlib.sha256(x).hexdigest() for x in braw],
            'design_weights':dict(zip(design_names,map(float,opt.x),strict=True)),
            'max_design_weight': args.max_design_weight,
            'objective':float(opt.fun),'effective_designs':float(1/np.sum(opt.x**2)),
            'positive_designs':int(np.count_nonzero(opt.x>1e-8)), 'statistics':statistics,
            'note':'Exploratory design sampling with equal rules per design; prior test matrix was inspected. Not a gate.'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('objective','effective_designs','positive_designs','statistics')},indent=2))


if __name__=='__main__':main()
