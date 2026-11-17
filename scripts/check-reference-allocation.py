"""Assess a frozen reference-specific allocation on another measurement draw."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from exposure.design_breadth import key, predicate_vocabulary, rule_log_breadth
from exposure.design_diagnostics import diagnose
from exposure.design_subspace import diagnose as subspace
from exposure.measure import load_rules


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--breadth',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=[x.read_bytes() for x in (a.frozen,a.report,a.breadth)]
    frozen,report,breadth=[json.loads(x) for x in raw]
    if hashlib.sha256(raw[1]).hexdigest() in frozen['fitting_report_sha256']:
        raise ValueError('Expected a new measurement report')
    if breadth['source_sha256']!=hashlib.sha256(raw[1]).hexdigest():
        raise ValueError('Breadth/source mismatch')
    names=frozen['rule_order'];rules=load_rules(report['rules']);atoms=predicate_vocabulary(rules,'role')
    if list(rules)!=names or breadth['predicate_keys']!=[key(x) for x in atoms]:
        raise ValueError('Rule or predicate mismatch')
    designs=frozen['designs'];diag=diagnose(report,{'designs':designs})
    if diag['excluded_count']:raise ValueError('Incomplete design rows')
    matrix=np.asarray(breadth['counts_by_condition'],dtype=float)/breadth['samples_per_condition']
    by_design={d['name']:rule_log_breadth(matrix,[d['weights'][n] for n in names],rules,atoms) for d in designs}
    rows=[{**r,'log_breadth':by_design[r['design']][r['rule']],
           'reference_group':float(r['design'].startswith('fitted_'))} for r in diag['rows']]
    if any(r['log_breadth'] is None for r in rows):raise ValueError('Undefined breadth')
    spec=importlib.util.spec_from_file_location('reference_allocation',Path(__file__).with_name('explore-reference-allocation.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    allocation=frozen['allocations'];weights=np.array([allocation[group][name] for group in ('uniform','fitted') for name in names])
    if np.any(weights<0) or not np.isclose(weights[:len(names)].sum(),.5) or not np.isclose(weights[len(names):].sum(),.5):
        raise ValueError('Invalid frozen allocation')
    stat=mod.statistics(*mod.moments(rows,names),weights)
    by_name={d['name']:d for d in designs};refs={g:by_name[g+'_reference']['weights'] for g in ('uniform','fitted')}
    single=np.asarray([x['column_probabilities'] for x in subspace(report,refs['uniform'])['singleton_events']]);drift={}
    for group,ref in refs.items():
        base=np.asarray([ref[n] for n in names]);v=[float(np.max(np.abs(single@(np.asarray([d['weights'][n] for n in names])-base))))
                                                   for d in designs if d['name'].startswith(group+'_')]
        drift[group]={'max':max(v),'median_max':float(np.median(v))}
    result={'schema_version':1,'input_sha256':[hashlib.sha256(x).hexdigest() for x in raw],
            'seed':report['seed'],'samples_per_condition':report['samples_per_condition'],
            'statistics':stat,'singleton_deviations':drift,
            'note':'Frozen exploratory group-specific allocation; different rule populations across reference groups. Not a gate.'}
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
