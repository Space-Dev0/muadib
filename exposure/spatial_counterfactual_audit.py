"""Audit checker spatial decisions on deterministic clean rendered counterfactuals.

This is a synthetic diagnostic, not a generated-image or human validation.
"""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

from exposure.checker import DetectionError, detect, evaluate_variable_count_experimental
from exposure.render import SHAPES, _draw
from exposure.rules import Attribute, Relation, SceneObject, satisfies

DELTAS = (-5, -2, -1, 0, 1, 2, 5)


def cases():
    """Keep objects separated orthogonally while shifting the blue object's axis."""
    for kind, delta, shapes in product(('left_of', 'above'), DELTAS, product(SHAPES, repeat=3)):
        if kind == 'left_of':
            positions = ((32, 12), (32 + delta, 32), (32, 52))
        else:
            positions = ((12, 32), (32, 32 + delta), (52, 32))
        objects = tuple(SceneObject(color, shape, *position) for color, shape, position
                        in zip(('red', 'blue', 'green'), shapes, positions, strict=True))
        rel = Relation('a', kind, 'b')
        yield kind, delta, shapes, objects, {
            'singleton': (rel,),
            'bound': (Attribute('a', 'color', 'red'), Attribute('b', 'color', 'blue'), rel),
        }


def run():
    groups = defaultdict(lambda: {'total': 0, 'positive_truth': 0, 'negative_truth': 0,
                                  'accepted': 0, 'abstained': 0, 'false_positive': 0,
                                  'false_negative': 0})
    mistakes = []
    for kind, delta, shapes, objects, rules in cases():
        image = _draw(objects).image
        try:
            detected = detect(image)
        except DetectionError:
            detected = None
        for rule_name, rule in rules.items():
            truth = satisfies(objects, rule)
            try:
                candidate = evaluate_variable_count_experimental(image, rule)
            except DetectionError:
                candidate = None
            standard = satisfies(detected, rule) if detected is not None else None
            for checker, result in (('standard', standard), ('variable_count', candidate)):
                key = (kind, delta, rule_name, checker)
                d = groups[key]
                d['total'] += 1
                d['positive_truth' if truth else 'negative_truth'] += 1
                d['accepted' if result is not None else 'abstained'] += 1
                if result is not None and result != truth:
                    error = 'false_positive' if result else 'false_negative'
                    d[error] += 1
                    mistakes.append({'kind': kind, 'delta': delta, 'shapes': list(shapes),
                                     'rule': rule_name, 'checker': checker, 'error': error})
    return {'schema_version': 1, 'cases': 2 * len(DELTAS) * len(SHAPES) ** 3,
            'groups': [{'kind': k[0], 'delta': k[1], 'rule': k[2], 'checker': k[3], **v}
                       for k, v in sorted(groups.items())],
            'mistakes': mistakes,
            'note': 'Clean synthetic images with known renderer annotations, not human review or generated-image validation. '
                    'Cases and paired counterfactuals are not independent random draws.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(f"Wrote {args.output}; {result['cases']} scenes; {len(result['mistakes'])} accepted disagreements")


if __name__ == '__main__':
    main()
