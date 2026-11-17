"""Compute the exposure matrix with shared assignment evaluation.

This is restricted to the existing three-object renderer and a/b/c variables.
It reproduces the original estimator's RNG stream and output semantics.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import uuid
from pathlib import Path

import numpy as np
from exposure.design_breadth import key
from exposure.measure import load_rules, unique_object, wilson_interval
from exposure.render import sample_conditioned
from exposure.rules import Attribute, _matches


def estimate(rules, *, samples, seed, max_attempts=10000, state_path=None, collect_breadth=False):
    if not rules or type(samples) is not int or samples < 1 or type(seed) is not int or max_attempts < 1:
        raise ValueError('Rules, samples and attempt limit must be nonempty and positive')
    atoms = sorted({p for predicates in rules.values() for p in predicates}, key=key)
    for atom in atoms:
        variables = (atom.variable,) if isinstance(atom, Attribute) else (atom.subject, atom.object)
        if any(v not in ('a', 'b', 'c') for v in variables):
            raise ValueError('Fast estimator only supports a/b/c variables')
    index = {p: i for i, p in enumerate(atoms)}
    names = list(rules); subsets = {}
    width = max(map(len, rules.values()))
    indices = np.zeros((len(names), width), dtype=int)
    active = np.zeros((len(names), width), dtype=bool)
    for row, (name, predicates) in enumerate(rules.items()):
        indices[row, :len(predicates)] = [index[p] for p in predicates]
        active[row, :len(predicates)] = True
        subsets[name] = [(i,) for i in range(len(predicates))] + list(itertools.combinations(range(len(predicates)), 2))
    rng = random.Random(seed); columns = []; breadth_counts = []
    rule_fingerprint = hashlib.sha256(json.dumps({name: [key(p) for p in ps] for name, ps in rules.items()},
                                                  separators=(',', ':')).encode()).hexdigest()
    from exposure import design_breadth, measure, render, rules as rules_module
    dependencies = (design_breadth, measure, render, rules_module)
    configuration = {'samples': samples, 'seed': seed, 'max_attempts': max_attempts,
                     'rules_sha256': rule_fingerprint,
                     'estimator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'dependency_sha256': {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                           for module in dependencies},
                     'python_version': list(sys.version_info[:3]), 'numpy_version': np.__version__}
    state_path = Path(state_path) if state_path is not None else None
    if state_path is not None and state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
        if state.get('configuration') != configuration:
            raise ValueError('Saved measurement state does not match configuration or code')
        columns = state['columns']; breadth_counts = state['breadth_counts']
        if (len(columns) != len(breadth_counts) or len(columns) > len(names) or
                [column['condition'] for column in columns] != names[:len(columns)]):
            raise ValueError('Saved measurement state has inconsistent columns')
        saved = state['rng_state']
        rng.setstate((saved[0], tuple(saved[1]), saved[2]))
    for condition, predicates in list(rules.items())[len(columns):]:
        counts = np.zeros(len(names), dtype=np.int64)
        pair_counts = np.zeros((len(atoms), len(atoms)), dtype=np.int64)
        attempts = 0
        for number in range(samples):
            try:
                scene, used = sample_conditioned(rng, predicates, max_attempts)
            except ValueError as exc:
                raise ValueError(f'Condition {condition!r}, sample {number + 1}: {exc}') from exc
            attempts += used
            if len(scene) != 3:
                raise ValueError('Fast estimator requires exactly three objects')
            truth = np.empty((6, len(atoms)), dtype=bool)
            for row, objects in enumerate(itertools.permutations(scene)):
                assignment = dict(zip(('a','b','c'), objects, strict=True))
                truth[row] = [_matches(atom, assignment) for atom in atoms]
            counts += np.any(np.all(truth[:, indices] | ~active[None, :, :], axis=2), axis=0)
            pair_counts += np.any(truth[:, :, None] & truth[:, None, :], axis=0)
        def record(count, certain):
            count = int(count)
            return {'successes': count, 'probability': count / samples,
                    'wilson_95': [1.0, 1.0] if certain else list(wilson_interval(count, samples))}
        measurements = {name: record(counts[i], name == condition) for i, name in enumerate(names)}
        measured_subsets = {}
        for name, predicates in rules.items():
            measured_subsets[name] = []
            for selected in subsets[name]:
                i = index[predicates[selected[0]]]; j = index[predicates[selected[-1]]]
                measured_subsets[name].append({'predicate_indices': list(selected),
                                               **record(pair_counts[i, j], name == condition)})
        columns.append({'condition':condition, 'accepted_scenes':samples, 'total_attempts':attempts,
                        'measurements':measurements, 'predicate_subsets':measured_subsets})
        breadth_counts.append(pair_counts.tolist())
        if state_path is not None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = state_path.with_name(state_path.name + '.' + uuid.uuid4().hex + '.tmp')
            with temporary.open('x', encoding='utf-8') as f:
                json.dump({'configuration': configuration, 'columns': columns, 'breadth_counts': breadth_counts,
                           'rng_state': rng.getstate()}, f, separators=(',', ':'), allow_nan=False)
                f.write('\n'); f.flush(); os.fsync(f.fileno())
            os.replace(temporary, state_path)
            print(f'Completed {len(columns)}/{len(names)} conditions: {condition}', flush=True)
    report = {'schema_version':1, 'seed':seed, 'samples_per_condition':samples, 'include_subsets':True,
            'max_attempts_per_scene':max_attempts, 'interpretation':'scene satisfaction given requested condition',
            'rule_order':names, 'columns':columns}
    if not collect_breadth:
        return report
    breadth = {'schema_version': 1, 'mode': 'role', 'source_seed': seed, 'samples_per_condition': samples,
               'rule_order': names, 'predicate_keys': [key(p) for p in atoms], 'counts_by_condition': breadth_counts,
               'note': 'Role-sensitive syntactic partner vocabulary, shared bindings; exploratory, not a gate.'}
    return report, breadth


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules', type=Path, required=True); parser.add_argument('--samples',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True); parser.add_argument('--max-attempts',type=int,default=10000)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--breadth-output',type=Path)
    parser.add_argument('--state',type=Path,help='New or compatible resumable state file'); args=parser.parse_args()
    try:
        paths = [path.resolve() for path in (args.output, args.breadth_output, args.state) if path is not None]
        if len(set(paths)) != len(paths):
            raise ValueError('Output, breadth and state paths must be distinct')
        if args.output.exists() or (args.breadth_output and args.breadth_output.exists()):
            raise ValueError('Output already exists; choose new output paths')
        raw=args.rules.read_bytes(); data=json.loads(raw,object_pairs_hook=unique_object)
        result=estimate(load_rules(data),samples=args.samples,seed=args.seed,max_attempts=args.max_attempts,
                        state_path=args.state,collect_breadth=bool(args.breadth_output))
        report, breadth = result if args.breadth_output else (result, None)
        report['rules']=data; report['estimator']='shared-assignment-v1'
        report['estimator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with args.output.open('x',encoding='utf-8') as f:
            json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
        if breadth is not None:
            breadth['source_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
            with args.breadth_output.open('x',encoding='utf-8') as f:
                json.dump(breadth,f,separators=(',', ':'),allow_nan=False); f.write('\n')
    except (OSError,ValueError,TypeError,KeyError) as exc:
        parser.error(str(exc))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
