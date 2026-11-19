"""Experimental bounded spatial relations; legacy rule semantics are unchanged.

Coordinates follow image convention: x increases right, y increases down.
A fixed-role relation is not the same event as an existential scene relation.
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from exposure.measure import wilson_interval
from exposure.render import _sample_objects
from exposure.rules import SceneObject


def sector_holds(subject: SceneObject, target: SceneObject, *, direction: str,
                 half_angle_degrees: float = 15., min_distance: float = 14.,
                 max_distance: float = 60.) -> bool:
    """Test subject's displacement from target inside an inclusive bounded cone."""
    if direction not in ('left','right','above','below'): raise ValueError('unknown direction')
    params=(half_angle_degrees,min_distance,max_distance)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in params):
        raise ValueError('sector parameters must be finite numbers')
    if not 0 < half_angle_degrees < 45 or not 0 < min_distance <= max_distance:
        raise ValueError('invalid angle or distance bounds')
    dx=subject.x-target.x;dy=subject.y-target.y; distance=math.hypot(dx,dy)
    if not min_distance <= distance <= max_distance:return False
    forward, lateral={'left':(-dx,dy),'right':(dx,dy),'above':(-dy,dx),'below':(dy,dx)}[direction]
    return forward > 0 and abs(lateral) <= forward*math.tan(math.radians(half_angle_degrees))


def exists_sector(objects, *, direction, subject_filter=None, target_filter=None, **kwargs):
    """Existential distinct-object sector event with optional endpoint filters."""
    def matches(obj, spec):
        return spec is None or all(getattr(obj,k)==v for k,v in spec.items())
    for spec in (subject_filter,target_filter):
        if spec is not None and (not isinstance(spec,dict) or set(spec)-{'color','shape'}):
            raise ValueError('filters may contain only color and shape')
    return any(i!=j and matches(a,subject_filter) and matches(b,target_filter)
               and sector_holds(a,b,direction=direction,**kwargs)
               for i,a in enumerate(objects) for j,b in enumerate(objects))


def simulate(samples,seed):
    if type(samples) is not int or samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError('samples must be positive and seed nonnegative')
    rng=random.Random(seed);counts={k:0 for k in ('fixed_left','existential_left','bound_left')}
    for _ in range(samples):
        objects=_sample_objects(rng)
        counts['fixed_left']+=sector_holds(objects[0],objects[1],direction='left')
        counts['existential_left']+=exists_sector(objects,direction='left')
        counts['bound_left']+=exists_sector(objects,direction='left',subject_filter={'color':'red','shape':'circle'},
                                            target_filter={'color':'blue','shape':'square'})
    return {'schema_version':1,'seed':seed,'samples':samples,
            'sector':{'direction':'left','half_angle_degrees':15,'min_distance':14,'max_distance':60},
            'counts':counts,'probabilities':{k:v/samples for k,v in counts.items()},
            'pointwise_wilson_95':{k:list(wilson_interval(v, samples)) for k,v in counts.items()},
            'note':'Base renderer simulation, exploratory fixed-role and existential events; not a checker or Stage 0 gate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',type=int,default=20000)
    p.add_argument('--seed',type=int,default=82810);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:
        result=simulate(a.samples,a.seed);a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(result,indent=2))
    except (ValueError,OSError) as exc:p.error(str(exc))
if __name__=='__main__':main()
