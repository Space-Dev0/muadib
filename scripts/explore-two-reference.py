import json, numpy as np
from exposure.design_reference_explore import matrix,optimize_reference
from exposure.design_extremes import generate
from exposure.design_diagnostics import diagnose
p=json.load(open('outputs/exposure-57-pilot.json')); h=json.load(open('outputs/exposure-57-holdout-81205.json'))
n,v,c=matrix(p); uniform=dict.fromkeys(n,1/len(n)); fitted=dict(zip(n,map(float,optimize_reference(v,c))))
all_design=[]
for prefix,weights in [('uniform',uniform),('fitted',fitted)]:
 for d in generate(p,weights,min_weight=.0001)['designs']:
  all_design.append({'name':prefix+'_'+d['name'],'weights':d['weights']})
for label,report in [('pilot',p),('holdout',h)]:
 r=diagnose(report,{'designs':all_design});rows=r['rows']; by={}
 for x in rows:by.setdefault(x['rule'],[]).append(x)
 within=np.array([[x['log_mean_marginal']-np.mean([y['log_mean_marginal'] for y in b]), x['log_lift']-np.mean([y['log_lift'] for y in b])] for b in by.values() for x in b]);print(label,'within_norm',float(np.linalg.norm(within[:,0])),'rank',np.linalg.matrix_rank(within),'r',float(np.corrcoef(within.T)[0,1]),'vif',r['vif'],'excluded',r['excluded_count'])
