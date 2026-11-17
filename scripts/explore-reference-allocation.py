"""Explore reference-dependent rule allocations; this changes the estimand."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from exposure.design_breadth import key, predicate_vocabulary, rule_log_breadth
from exposure.design_diagnostics import FEATURES, diagnose
from exposure.measure import load_rules

FEATURE_NAMES = (*FEATURES, 'log_breadth', 'reference_group')


def equal_rule_constraints(n):
    """One marginal rule distribution and equal total reference-group mass."""
    if type(n) is not int or n < 1:
        raise ValueError('Expected a positive rule count')
    eq = np.zeros((n+1,2*n)); eq[:n,:n]=np.eye(n); eq[:n,n:]=np.eye(n); eq[n,:n]=1
    return eq, np.r_[np.full(n,1/n), .5]


def moments(rows, names):
    groups = {(group, name): [] for group in ('uniform', 'fitted') for name in names}
    for row in rows:
        group = 'fitted' if row['design'].startswith('fitted_') else 'uniform'
        groups[(group, row['rule'])].append([row[f] for f in FEATURE_NAMES])
    if not all(groups.values()) or len({len(x) for x in groups.values()}) != 1:
        raise ValueError('Incomplete reference/rule groups')
    arrays = [np.asarray(groups[(group, name)], dtype=float)
              for group in ('uniform', 'fitted') for name in names]
    return (np.stack([a.mean(axis=0) for a in arrays]),
            np.stack([a.T @ a / len(a) for a in arrays]))


def statistics(means, seconds, weights):
    w = np.asarray(weights, dtype=float); w = w / w.sum(); mean = w @ means
    covariance = np.einsum('i,ijk->jk', w, seconds) - np.outer(mean, mean)
    scale = np.sqrt(np.diag(covariance))
    if not np.isfinite(scale).all() or np.any(scale < 1e-12):
        raise ValueError('Invalid covariance')
    corr = covariance / np.outer(scale, scale)
    if np.linalg.eigvalsh(corr).min() <= 1e-10:
        raise ValueError('Collinear features')
    return {'marginal_lift_r': float(corr[0,1]),
            'vif': dict(zip(FEATURE_NAMES, map(float, np.diag(np.linalg.inv(corr))), strict=True))}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoints',default='outputs/design-breadth-multimatrix-82531.json')
    p.add_argument('--reports',nargs=3,default=['outputs/exposure-57-fresh-82505.json',
                   'outputs/exposure-57-fresh-82501.json','outputs/exposure-57-fresh-82507.json'])
    p.add_argument('--breadths',nargs=3,default=['outputs/breadth-57-fresh-82505.json',
                   'outputs/breadth-57-fresh-82501.json','outputs/breadth-57-fresh-82507.json'])
    p.add_argument('--min-multiple',type=float,default=.05); p.add_argument('--max-multiple',type=float,default=4.)
    p.add_argument('--equal-overall-rules',action='store_true')
    args=p.parse_args()
    if not 0 < args.min_multiple <= 1 <= args.max_multiple:
        raise ValueError('Invalid allocation bounds')
    design_raw = Path(args.endpoints).read_bytes(); frozen=json.loads(design_raw)
    report_paths=args.reports; breadth_paths=args.breadths
    report_raw=[Path(x).read_bytes() for x in report_paths]; breadth_raw=[Path(x).read_bytes() for x in breadth_paths]
    reports=[json.loads(x) for x in report_raw]; breadths=[json.loads(x) for x in breadth_raw]
    rules=load_rules(reports[0]['rules']); names=list(rules); atoms=predicate_vocabulary(rules,'role'); designs=frozen['designs']
    result_moments=[]
    for raw,report,b in zip(report_raw,reports,breadths,strict=True):
        if hashlib.sha256(raw).hexdigest()!=b['source_sha256'] or b['predicate_keys']!=[key(x) for x in atoms]:
            raise ValueError('Breadth/source mismatch')
        if load_rules(report['rules'])!=rules:
            raise ValueError('Rules differ')
        diagnostic=diagnose(report,{'designs':designs})
        if diagnostic['excluded_count']:
            raise ValueError('Incomplete rows')
        matrix=np.asarray(b['counts_by_condition'],dtype=float)/b['samples_per_condition']
        bd={d['name']:rule_log_breadth(matrix,[d['weights'][name] for name in names],rules,atoms) for d in designs}
        rows=[{**row,'log_breadth':bd[row['design']][row['rule']],
               'reference_group':float(row['design'].startswith('fitted_'))} for row in diagnostic['rows']]
        if any(row['log_breadth'] is None for row in rows):
            raise ValueError('Undefined breadth')
        result_moments.append(moments(rows,names))
    n=len(names); initial=np.array([frozen['rule_allocation'][name]/2 for _ in range(2) for name in names])
    def objective(w):
        try:
            results=[statistics(*m,w) for m in result_moments[:2]]
            return sum(float(np.sum(np.maximum(np.array(list(r['vif'].values()))-1.85,0)**2))
                       +25*max(abs(r['marginal_lift_r'])-.24,0)**2 for r in results)
        except (ValueError,np.linalg.LinAlgError): return 1e6
    if args.equal_overall_rules:
        spec=importlib.util.spec_from_file_location('equal_designs',Path(__file__).with_name('explore-equal-rule-designs.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        eq,target=equal_rule_constraints(n)
        initial=np.full(2*n,1/(2*n))
        fit=minimize(lambda w:module.objective_gradient(w,result_moments[:2]),initial,jac=True,method='SLSQP',
                     bounds=[(args.min_multiple/(2*n),args.max_multiple/(2*n))]*(2*n),
                     constraints={'type':'eq','fun':lambda w:eq@w-target,'jac':lambda w:eq},
                     options={'maxiter':1000,'ftol':1e-10})
    else:
        fit=minimize(objective,initial,method='SLSQP',bounds=[(args.min_multiple/(2*n),args.max_multiple/(2*n))]*(2*n),
                     constraints=[{'type':'eq','fun':lambda w:w[:n].sum()-.5},
                                  {'type':'eq','fun':lambda w:w[n:].sum()-.5}],
                     options={'maxiter':1000,'ftol':1e-10})
    if not fit.success or abs(fit.x[:n].sum()-.5)>1e-8 or abs(fit.x[n:].sum()-.5)>1e-8:
        raise ValueError(f'Allocation failed: {fit.message}')
    if args.equal_overall_rules and np.max(np.abs(fit.x[:n] + fit.x[n:] - 1/n)) > 1e-8:
        raise ValueError('Equal overall rule representation was not achieved')
    allocations={group:dict(zip(names,map(float,fit.x[i*n:(i+1)*n]),strict=True))
                 for i,group in enumerate(('uniform','fitted'))}
    output={'schema_version':1,'endpoint_source_sha256':hashlib.sha256(design_raw).hexdigest(),
            'report_sha256':[hashlib.sha256(x).hexdigest() for x in report_raw],
            'breadth_sha256':[hashlib.sha256(x).hexdigest() for x in breadth_raw],
            'allocations':allocations,'objective':float(fit.fun),
            'bounds_relative_to_equal': [args.min_multiple,args.max_multiple],
            'equal_overall_rules': args.equal_overall_rules,
            'effective_rules_by_group':{group:float(1/np.sum((fit.x[i*n:(i+1)*n]*2)**2))
                                        for i,group in enumerate(('uniform','fitted'))},
            'statistics':[statistics(*m,fit.x) for m in result_moments],
            'note':'Exploratory post-hoc group-specific allocation; rule population differs between references. Not a gate.'}
    with args.output.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:output[k] for k in ('objective','effective_rules_by_group','statistics')},indent=2))


if __name__=='__main__':main()
