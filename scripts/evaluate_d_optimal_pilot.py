"""Reproduce exploratory between-rule D-optimal allocation on saved artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from exposure.design_diagnostics import diagnose
from exposure.design_optimal import solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--designs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw, d_raw = args.report.read_bytes(), args.designs.read_bytes()
    report, design_data = json.loads(raw), json.loads(d_raw)
    designs = design_data['designs'][:3]
    rows = [row for row in diagnose(report, {'designs': designs})['rows']
            if row['design'] == 'uniform_reference']
    names = ['log_mean_marginal', 'log_lift', 'k_attr', 'k_rel']
    matrix = np.array([[row[key] for key in names] for row in rows])
    scale = matrix.std(axis=0)
    if np.any(scale == 0):
        raise ValueError('constant feature is not identified')
    z = (matrix - matrix.mean(axis=0)) / scale
    x = np.column_stack([np.ones(len(z)), z]); n = len(x)
    constraints = [{'coefficients':np.eye(n)[i].tolist(), 'max':4/n} for i in range(n)]
    result = solve(x, constraints=constraints, min_weight=.05/n)
    weights = np.array(result['weights']); uniform = np.full(n, 1/n)
    def vif(q):
        centered = z - q @ z
        covariance = centered.T @ (q[:,None] * centered)
        return (np.diag(covariance) * np.diag(np.linalg.inv(covariance))).tolist()
    result.update({'note':'Exploratory between-rule weighting of one fixed reference; not a comparison of trained models or within-rule identification.',
                   'report_sha256':hashlib.sha256(raw).hexdigest(),
                   'designs_sha256':hashlib.sha256(d_raw).hexdigest(),
                   'feature_names':['intercept']+names, 'rule_order':[r['rule'] for r in rows],
                   'uniform_logdet':float(np.linalg.slogdet(x.T@(uniform[:,None]*x))[1]),
                   'effective_rules':float(1/(weights@weights)),
                   'uniform_vif':vif(uniform), 'optimized_vif':vif(weights)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k:result[k] for k in ('uniform_logdet','log_determinant','effective_rules','uniform_vif','optimized_vif')}))

if __name__ == '__main__':
    main()
