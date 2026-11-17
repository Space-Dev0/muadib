"""Evaluate frozen global mixture weights on other estimated exposure matrices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from exposure.design_factorial import contrasts, features, load_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design', type=Path, required=True)
    p.add_argument('--reports', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = args.design.read_bytes()
    design = json.loads(raw)
    if not design.get('best') or not design['best'].get('feasible'):
        raise ValueError('design has no accepted fitting solution')
    weights = np.asarray(design['best']['weights'], dtype=float)
    s = design['settings']; rows=[]
    for path in args.reports:
        report_raw = path.read_bytes(); report = json.loads(report_raw)
        order, marginal, joint = load_matrix(report, design['rule_names'])
        if order != design['condition_order'] or weights.shape != (4, len(order)):
            raise ValueError('condition order or weight shape mismatch')
        x, _ = features(weights, marginal, joint)
        centered = x - x.mean(axis=0, keepdims=True)
        eigenvalues = np.linalg.eigvalsh(np.einsum('kri,krj->ij', centered, centered))
        margins = contrasts(weights.ravel(), marginal, joint, s['delta_m'], s['delta_l'], s['cross_tolerance'])[0]
        rows.append({'report_sha256':hashlib.sha256(report_raw).hexdigest(), 'seed':report['seed'],
                     'samples_per_condition':report['samples_per_condition'],
                     'information_eigenvalues':eigenvalues.tolist(),
                     'information_condition_number':float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 1e-12 else None,
                     'worst_constraint_margin':float(margins.min()),
                     'number_violations':int(np.count_nonzero(margins < -1e-7))})
    result={'design_sha256':hashlib.sha256(raw).hexdigest(), 'reports':rows,
            'max_condition_weight':float(weights.max()),
            'effective_conditions':(1 / np.square(weights).sum(axis=1)).tolist(),
            'note':'Interpret each draw using its pre-specification and prior inspection history; no uncertainty or Stage 0 claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out: json.dump(result, out, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
