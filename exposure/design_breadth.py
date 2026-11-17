"""Exploratory role-sensitive predicate co-occurrence breadth.

The partner universe is the distinct syntactic atomic predicates in the given
rule table. Shared variables bind to the same object and different variables to
 distinct objects. This arbitrary role universe and geometric rule aggregation
are research choices, not a registered proposal gate.
"""
import argparse
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
from exposure.measure import load_rules, unique_object
from exposure.render import sample_conditioned
from exposure.rules import Attribute, Relation, _matches


def key(predicate):
    return json.dumps({'type': 'attribute' if isinstance(predicate, Attribute) else 'relation',
                       **asdict(predicate)}, sort_keys=True, separators=(',', ':'))


def pair_matrix(scene, predicates):
    """Existential co-satisfaction with shared named-object bindings.

    All atomic variables must be among a,b,c and each scene has three objects;
    every partial distinct binding extends to a permutation of all three.
    """
    if len(scene) != 3:
        raise ValueError('Expected exactly three scene objects')
    matched = np.zeros((len(predicates), len(predicates)), dtype=bool)
    for objects in itertools.permutations(scene):
        assignment = dict(zip(('a', 'b', 'c'), objects, strict=True))
        values = np.fromiter((_matches(p, assignment) for p in predicates), bool, count=len(predicates))
        matched |= values[:, None] & values[None, :]
    return matched


def estimate(report):
    rules = load_rules(report['rules'])
    if list(rules) != report['rule_order']:
        raise ValueError('Rule order does not match report')
    samples, seed = report['samples_per_condition'], report['seed']
    if type(samples) is not int or samples < 1 or type(seed) is not int:
        raise ValueError('Invalid report sampling configuration')
    predicates = sorted({p for ps in rules.values() for p in ps}, key=key)
    indices = {p: i for i, p in enumerate(predicates)}
    for p in predicates:
        variables = (p.variable,) if isinstance(p, Attribute) else (p.subject, p.object)
        if any(v not in ('a', 'b', 'c') for v in variables):
            raise ValueError('Only variables a, b and c are supported')
    rng = random.Random(seed); columns = []
    if len(report['columns']) != len(rules) or not report.get('include_subsets'):
        raise ValueError('Expected complete source columns with predicate subsets')
    for source_column, (name, condition) in zip(report['columns'], rules.items(), strict=True):
        if source_column['condition'] != name:
            raise ValueError('Source columns are out of order')
        counts = np.zeros((len(predicates), len(predicates)), dtype=np.int64)
        for _ in range(samples):
            scene, _ = sample_conditioned(rng, condition, report['max_attempts_per_scene'])
            counts += pair_matrix(scene, predicates)
        for rule_name, atoms in rules.items():
            for subset in source_column['predicate_subsets'][rule_name]:
                selected = subset['predicate_indices']
                a = indices[atoms[selected[0]]]
                b = indices[atoms[selected[-1]]]
                if int(counts[a, b]) != subset['successes']:
                    raise ValueError(f'Regenerated scene counts do not match source: {name}, {rule_name}')
        columns.append(counts.tolist())
    return {'schema_version': 1, 'source_seed': seed, 'samples_per_condition': samples,
            'rule_order': list(rules), 'predicate_keys': [key(p) for p in predicates],
            'counts_by_condition': columns,
            'note': ('Role-sensitive syntactic partner vocabulary; equal weight per partner event; '
                     'a scene can contribute multiple partner events. Rule breadth is the geometric '
                     'mean of atomic breadth. Exploratory, not a validated gate.')}


def rule_log_breadth(matrix, weights, rules, predicates):
    """Return log geometric-mean effective partner count per rule."""
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or len(weights) != len(matrix) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError('Invalid mixture weights')
    joint = np.einsum('i,ijk->jk', weights, np.asarray(matrix, dtype=float))
    np.fill_diagonal(joint, 0)
    lookup = {key(p): i for i, p in enumerate(predicates)}
    entropy = []
    for row in joint:
        total = row.sum()
        if total <= 0:
            entropy.append(None); continue
        probs = row[row > 0] / total
        entropy.append(float(-np.sum(probs * np.log(probs))))
    result = {}
    for name, atoms in rules.items():
        values = [entropy[lookup[key(p)]] for p in atoms]
        result[name] = None if any(x is None for x in values) else math.fsum(values) / len(values)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        raw = args.report.read_bytes(); result = estimate(json.loads(raw, object_pairs_hook=unique_object))
        result['source_sha256'] = hashlib.sha256(raw).hexdigest()
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, separators=(',', ':'), allow_nan=False); f.write('\n')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        p.error(str(exc))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
