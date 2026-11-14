import json,numpy as np
from exposure.design_reference_explore import matrix,optimize_reference
from exposure.design_extremes import generate
from exposure.design_diagnostics import diagnose
from exposure.design_rule_allocation import moments,fit,statistics
p=json.load(open('outputs/exposure-57-pilot.json'));h=json.load(open('outputs/exposure-57-holdout-81205.json'))
n,v,c=matrix(p); design=[]
for prefix,w in [('uniform',dict.fromkeys(n,1/len(n))),('fitted',dict(zip(n,map(float,optimize_reference(v,c)))) )]:
 for d in generate(p,w,min_weight=.0001)['designs']:design.append({'name':prefix+'_'+d['name'],'weights':d['weights']})
a,b=[moments(diagnose(x,{'designs':design})['rows'],n) for x in (p,h)]
q,obj=fit(*a);print('fit',statistics(*a,q),'holdout',statistics(*b,q),'min',min(q),'max',max(q),'effective',1/sum(q*q))
