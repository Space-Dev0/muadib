"""Generate exploratory, factorial two-endpoint-bound relational rules.

The split is at directed endpoint attribute-pair level, not a split of images.
This does not solve existential relation-singleton saturation or the renderer's
sampling, model, checker, exposure-design, and human-review gates.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from exposure.measure import load_rules
from exposure.render import COLORS, SHAPES
from exposure.templates import canonical_key


def build():
    atoms = [(color, shape) for color in COLORS for shape in SHAPES]
    rules, split = {}, {'train': [], 'heldout': []}
    for i, (subject_color, subject_shape) in enumerate(atoms):
        for j, (object_color, object_shape) in enumerate(atoms):
            # Both relation kinds for a pair stay together: avoids putting the
            # same endpoint configuration in both partitions via relation kind.
            partition = 'heldout' if (i + j) % 4 == 0 else 'train'
            for kind in ('left_of', 'above'):
                name = f'{subject_color}_{subject_shape}_{kind}_{object_color}_{object_shape}'
                rules[name] = [
                    {'type':'attribute', 'variable':'a', 'field':'color', 'value':subject_color},
                    {'type':'attribute', 'variable':'a', 'field':'shape', 'value':subject_shape},
                    {'type':'attribute', 'variable':'b', 'field':'color', 'value':object_color},
                    {'type':'attribute', 'variable':'b', 'field':'shape', 'value':object_shape},
                    {'type':'relation', 'subject':'a', 'kind':kind, 'object':'b'},
                ]
                split[partition].append(name)
    parsed = load_rules(rules)
    keys = [canonical_key(p) for p in parsed.values()]
    if len(set(keys)) != len(keys):
        raise RuntimeError('Generated semantically duplicate templates')
    coverage = {}
    for part, names in split.items():
        counts = Counter()
        for name in names:
            preds = parsed[name]
            counts['left_of' if name.find('_left_of_') >= 0 else 'above'] += 1
            for variable in ('a', 'b'):
                values = {p.field: p.value for p in preds if getattr(p, 'variable', None) == variable}
                counts['pair:' + values['color'] + '/' + values['shape']] += 1
        coverage[part] = dict(sorted(counts.items()))
    return rules, {
        'schema_version':1,
        'description':'Exploratory factorial endpoint-bound templates; not a validated benchmark.',
        'split_unit':'directed subject/object attribute-pair; both relation kinds assigned together',
        'train':split['train'], 'heldout':split['heldout'], 'coverage':coverage,
        'limitations':[
            'Held-out template metadata does not establish independent scene, data, or model splits.',
            'The current existential single-relation event remains near-tautological under the renderer.',
            'Pairwise balanced attributes do not establish sampling support or equal rule prevalence.',
            'No checker, exposure-design, or model-learning validation is implied.',
            'Supply this manifest explicitly to benchmark_audit; the split does not establish actual use.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules-output', type=Path, required=True)
    parser.add_argument('--manifest-output', type=Path, required=True)
    args = parser.parse_args()
    rules_path, manifest_path = args.rules_output.resolve(), args.manifest_output.resolve()
    if rules_path == manifest_path or rules_path.exists() or manifest_path.exists():
        parser.error('outputs must be distinct and must not already exist')
    rules, manifest = build()
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting existing experiment artifacts.
    with rules_path.open('x', encoding='utf-8') as f:
        json.dump(rules, f, indent=2); f.write('\n')
    with manifest_path.open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2); f.write('\n')

if __name__ == '__main__':
    main()
