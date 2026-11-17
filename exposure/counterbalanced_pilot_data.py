"""Create paired-complexity, counterbalanced synthetic scenes for future studies.

This does not train a model, repair spatial predicates, or pass Stage 0. The
optional holdout excludes chosen color/shape combinations from every object in
training; the ordinary balanced design includes all combinations in training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path

from exposure.render import COLORS, SHAPES, _draw
from exposure.rules import SceneObject

VOCABULARY = tuple((color, shape) for color in COLORS for shape in SHAPES)
GRID = tuple((x, y) for y in (12, 32, 52) for x in (12, 32, 52))


def generate(output: Path, *, seed: int, train_pairs: int, test_pairs: int,
             heldout_combinations=()) -> dict:
    """Produce independent split pairs; each high scene adds one object to low.

    Training and test counts must be balanced over their respective target
    vocabularies. Exact image duplicates across splits are rejected.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    holdout = tuple(tuple(item) for item in heldout_combinations)
    if len(set(holdout)) != len(holdout) or any(pair not in VOCABULARY for pair in holdout):
        raise ValueError('holdout combinations must be unique valid color/shape pairs')
    train_vocab = tuple(pair for pair in VOCABULARY if pair not in set(holdout))
    if len(train_vocab) < 3:
        raise ValueError('at least three train combinations must remain')
    for name, count, unit in [('train_pairs', train_pairs, len(train_vocab)),
                              ('test_pairs', test_pairs, len(VOCABULARY))]:
        if type(count) is not int or count < unit or count % unit:
            raise ValueError(f'{name} must be a positive multiple of {unit}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    seen_by_split = {'train': set(), 'test': set()}
    records = []
    for split, count, vocab in [('train', train_pairs, train_vocab), ('test', test_pairs, VOCABULARY)]:
        targets = list(vocab) * (count // len(vocab)); rng.shuffle(targets)
        for index, target in enumerate(targets):
            # Per-target role rotations give exact per-role balance when all
            # target combinations occur equally often within this split.
            i = vocab.index(target)
            labels = (target, vocab[(i + 1) % len(vocab)], vocab[(i + 2) % len(vocab)])
            for _ in range(1000):
                selected = rng.sample(GRID, 3)
                objects = tuple(SceneObject(color, shape, x + rng.randint(-3, 3), y + rng.randint(-3, 3))
                                for (color, shape), (x, y) in zip(labels, selected, strict=True))
                low, high = _draw(objects[:2]), _draw(objects)
                low_hash = hashlib.sha256(low.image.tobytes()).hexdigest()
                high_hash = hashlib.sha256(high.image.tobytes()).hexdigest()
                if low_hash != high_hash and low_hash not in seen_by_split['train'] | seen_by_split['test'] and high_hash not in seen_by_split['train'] | seen_by_split['test']:
                    break
            else:
                raise RuntimeError('unable to generate a unique pair after 1000 attempts')
            pair_id = f'{split}-{index:06d}'
            for complexity, rendered, digest in [('low', low, low_hash), ('high', high, high_hash)]:
                name = f'{pair_id}-{complexity}.png'
                rendered.image.save(output / name)
                seen_by_split[split].add(digest)
                records.append({'pair_id':pair_id, 'split':split, 'complexity':complexity,
                                'image':name, 'pixel_sha256':digest,
                                'target':{'color':target[0], 'shape':target[1]},
                                'objects':[asdict(obj) for obj in rendered.objects]})
    manifest = {'schema_version':1, 'seed':seed, 'train_pairs':train_pairs, 'test_pairs':test_pairs,
                'heldout_combinations':[list(pair) for pair in holdout],
                'counts':{split:dict(sorted(Counter(r['target']['color'] + '/' + r['target']['shape']
                          for r in records if r['split']==split and r['complexity']=='low').items()))
                          for split in ('train','test')}, 'records':records,
                'note':('Geometry-paired synthetic candidate; no model, exposure intervention, '
                        'independent checker validation or Stage 0 pass. Exact pixel uniqueness does not rule out near duplicates.')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=82700)
    p.add_argument('--train-pairs',type=int,default=120);p.add_argument('--test-pairs',type=int,default=24)
    p.add_argument('--holdout',action='append',default=[],help='optional COLOR:SHAPE; exclude from all training objects')
    a=p.parse_args()
    try:
        if any(item.count(':') != 1 for item in a.holdout): raise ValueError('holdout must use COLOR:SHAPE')
        result=generate(a.output,seed=a.seed,train_pairs=a.train_pairs,test_pairs=a.test_pairs,
                        heldout_combinations=[tuple(item.split(':')) for item in a.holdout])
    except (ValueError,OSError,RuntimeError) as exc: p.error(str(exc))
    print(json.dumps({'pairs':len(result['records'])//2,'output':str(a.output),'note':result['note']}))

if __name__ == '__main__': main()
