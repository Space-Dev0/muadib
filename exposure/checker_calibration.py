"""Synthetic, split-aware hard-negative calibration for existing pixel checkers.

Labels are renderer metadata, not human judgments; perturbed targets may be
ambiguous to people. This does not validate generalization to model outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from exposure import checker as checker_module
from exposure.checker import DetectionError, detect, evaluate_variable_count_experimental
from exposure.render import COLORS, SHAPES, _draw, parse_rule
from exposure.rules import SceneObject, satisfies

SCENARIOS=('present_distinct','present_competing','swapped_binding','missing_shape')
PERTURBATIONS=('clean','blur_0_5','blur_1','brightness_1_1','noise_10')
GRID=tuple((x,y) for y in (12,32,52) for x in (12,32,52))


def _sha(data): return hashlib.sha256(data).hexdigest()


def _perturb(image, name, seed):
    if name == 'clean': return image
    if name == 'blur_0_5': return image.filter(ImageFilter.GaussianBlur(.5))
    if name == 'blur_1': return image.filter(ImageFilter.GaussianBlur(1.0))
    if name == 'brightness_1_1': return ImageEnhance.Brightness(image).enhance(1.1)
    if name == 'noise_10':
        arr=np.asarray(image,dtype=np.float32)
        noise=np.random.default_rng(seed).normal(0,10,arr.shape)
        return Image.fromarray(np.clip(arr+noise,0,255).astype(np.uint8),'RGB')
    raise ValueError('unknown perturbation')


def _rule(color, shape):
    return [{'type':'attribute','variable':'a','field':'color','value':color},
            {'type':'attribute','variable':'a','field':'shape','value':shape}]


def _objects(rng, scenario, color, shape):
    other_color=rng.choice([c for c in COLORS if c!=color])
    other_shape=rng.choice([s for s in SHAPES if s!=shape])
    if scenario == 'present_distinct':
        labels=[(color,shape)] + [(rng.choice([c for c in COLORS if c!=color]),
                                   rng.choice([s for s in SHAPES if s!=shape])) for _ in range(2)]
    elif scenario == 'present_competing': labels=[(color,shape),(color,other_shape),(other_color,shape)]
    elif scenario == 'swapped_binding': labels=[(color,other_shape),(other_color,shape),(other_color,other_shape)]
    elif scenario == 'missing_shape': labels=[(rng.choice(tuple(COLORS)),rng.choice([s for s in SHAPES if s!=shape])) for _ in range(3)]
    else: raise ValueError('unknown scenario')
    rng.shuffle(labels)
    return tuple(SceneObject(c,s,x+rng.randint(-3,3),y+rng.randint(-3,3))
                 for (c,s),(x,y) in zip(labels,rng.sample(GRID,3),strict=True))


def generate(output: Path, *, seed=82800, train_per_stratum=15, validation_per_stratum=5, test_per_stratum=5):
    if type(seed) is not int or seed < 0: raise ValueError('seed must be nonnegative')
    if any(type(n) is not int or n < 1 for n in (train_per_stratum,validation_per_stratum,test_per_stratum)):
        raise ValueError('split counts must be positive integers')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    rng=random.Random(seed); records=[]; seen=set()
    vocabulary=[(c,s) for c in COLORS for s in SHAPES]
    for split,n in [('train',train_per_stratum),('validation',validation_per_stratum),('test',test_per_stratum)]:
        for scenario in SCENARIOS:
            for perturbation in PERTURBATIONS:
                targets=[]
                while len(targets)<n:
                    targets.extend(rng.sample(vocabulary, len(vocabulary)))
                targets=targets[:n]
                for color,shape in targets:
                    for _ in range(1000):
                        objs=_objects(rng,scenario,color,shape); rule=_rule(color,shape)
                        expected=scenario.startswith('present_')
                        if satisfies(objs,parse_rule(rule)) != expected: raise RuntimeError('scenario label mismatch')
                        clean=_draw(objs).image
                        clean_sha=_sha(clean.tobytes())
                        # Avoid identical clean scenes even when perturbations differ.
                        if clean_sha not in seen: break
                    else: raise RuntimeError('unable to find unique scene')
                    seen.add(clean_sha)
                    perturb_seed=rng.getrandbits(63)
                    image=_perturb(clean,perturbation,perturb_seed)
                    name=f'{len(records):06d}.png';image.save(output/name)
                    records.append({'image':name,'split':split,'scenario':scenario,'perturbation':perturbation,
                                    'perturb_seed':perturb_seed,'rule':rule,'objects':[asdict(o) for o in objs],
                                    'truth':expected,'clean_pixel_sha256':clean_sha,
                                    'pixel_sha256':_sha(image.tobytes()),'file_sha256':_sha((output/name).read_bytes())})
    m={'schema_version':1,'seed':seed,'records':records,
       'scope':'Metadata-labeled synthetic scenes only; no human ambiguity assessment or generated-image validation.'}
    (output/'manifest.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
    return m


def evaluate(dataset: Path, *, expected_sha256: str, split='test'):
    data=(dataset/'manifest.json').read_bytes()
    if _sha(data)!=expected_sha256: raise ValueError('manifest hash mismatch')
    manifest=json.loads(data); rows=[r for r in manifest['records'] if r['split']==split]
    if not rows: raise ValueError('split is empty')
    counts={method:defaultdict(Counter) for method in ('standard','experimental')}
    details=[]; seen=set()
    for r in rows:
        name=r['image']; path=dataset/name
        if Path(name).name!=name or name in seen: raise ValueError('invalid or duplicate image')
        seen.add(name)
        if _sha(path.read_bytes())!=r['file_sha256']: raise ValueError('image file hash mismatch')
        with Image.open(path) as opened: image=opened.convert('RGB')
        if _sha(image.tobytes())!=r['pixel_sha256']: raise ValueError('image pixel hash mismatch')
        predicates=parse_rule(r['rule']); objects=tuple(SceneObject(**o) for o in r['objects'])
        if type(r['truth']) is not bool or satisfies(objects,predicates)!=r['truth']:
            raise ValueError('truth does not match metadata')
        decisions={}
        for method in counts:
            try:
                result=(satisfies(detect(image),predicates) if method=='standard'
                        else evaluate_variable_count_experimental(image,predicates))
            except DetectionError: result=None
            if result is not None and type(result) is not bool: raise ValueError('invalid checker decision')
            label='abstain' if result is None else ('yes' if result else 'no')
            truth='positive' if r['truth'] else 'negative'
            for key in ('all',f"scenario:{r['scenario']}",f"perturbation:{r['perturbation']}"):
                for t in ('positive','negative'):
                    for decision in ('yes','no','abstain'):
                        counts[method][key].setdefault(t+'_'+decision,0)
                counts[method][key][truth+'_'+label]+=1
            decisions[method]=label
        details.append({'image':name,'truth':r['truth'],'scenario':r['scenario'],
                        'perturbation':r['perturbation'],'decisions':decisions})
    return {'schema_version':1,'dataset_manifest_sha256':expected_sha256,'split':split,'samples':len(rows),
            'checker_source_sha256':_sha(Path(checker_module.__file__).read_bytes()),
            'evaluation_source_sha256':_sha(Path(__file__).read_bytes()),
            'counts':{method:{group:dict(v) for group,v in groups.items()} for method,groups in counts.items()},
            'details':details,'scope':manifest['scope']}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    g=sub.add_parser('generate');g.add_argument('--output',type=Path,required=True);g.add_argument('--seed',type=int,default=82800)
    e=sub.add_parser('evaluate');e.add_argument('--dataset',type=Path,required=True);e.add_argument('--expected-sha256',required=True);e.add_argument('--split',default='test');e.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.mode=='generate':
            m=generate(a.output,seed=a.seed);print(json.dumps({'samples':len(m['records']),'manifest_sha256':_sha((a.output/'manifest.json').read_bytes())}))
        else:
            result=evaluate(a.dataset,expected_sha256=a.expected_sha256,split=a.split)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
            print(json.dumps({'samples':result['samples'],'counts':{k:v['all'] for k,v in result['counts'].items()}},indent=2))
    except (ValueError,OSError,KeyError,RuntimeError) as exc:p.error(str(exc))

if __name__=='__main__':main()
