"""Freeze experimental checker decisions before the fresh human review."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from exposure.checker import DetectionError, evaluate_variable_count_experimental
from exposure.rules import Attribute


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def predictions(pack):
    private = Path(pack) / 'private'
    provenance_bytes = (private / 'provenance.json').read_bytes()
    provenance = json.loads(provenance_bytes)
    if provenance['checkpoint_sha256'] != '84dac304a3a2b67ca8342f331b05adea940da746293dd7a3002bb902c43d91d4':
        raise ValueError('Unexpected checkpoint')
    expected = {(color, seed, 1.0) for color in ('red', 'blue') for seed in range(81530, 81546)}
    items = provenance['items']
    actual = [(row['color'], row['seed'], row['guidance']) for row in items.values()]
    if len(actual) != 32 or set(actual) != expected:
        raise ValueError('Pack does not contain the prespecified 32 samples')
    results = {}
    for name, item in sorted(items.items()):
        if Path(name).name != name:
            raise ValueError('Invalid sample ID')
        raw = (private / 'images' / f'{name}.png').read_bytes()
        if digest(raw) != item['image_sha256']:
            raise ValueError(f'Image hash mismatch: {name}')
        rule = (Attribute('x', 'color', item['color']), Attribute('x', 'shape', 'circle'))
        try:
            with Image.open(private / 'images' / f'{name}.png') as image:
                value = evaluate_variable_count_experimental(image, rule)
            reason = 'uncertain' if value is None else None
        except DetectionError as exc:
            value, reason = None, str(exc)
        results[name] = {'target_circle': value, 'abstention_reason': reason}
    from exposure import checker
    return {'schema_version': 1, 'pack_provenance_sha256': digest(provenance_bytes),
            'checker_sha256': digest(Path(checker.__file__).read_bytes()),
            'results': results, 'note': 'Experimental positive-witness checker; null denotes abstention. '
            'Decisions are not ground truth and do not evaluate object count or the Stage 0 gate.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        result = predictions(a.pack)
        with a.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(f'Wrote {a.output}')


if __name__ == '__main__':
    main()
