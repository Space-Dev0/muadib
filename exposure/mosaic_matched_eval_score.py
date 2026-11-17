"""Summarize matched-model reviews without resolving disputed or unsure answers."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from exposure.mosaic_guidance_review_score import decision
from exposure.mosaic_matched_eval_pack import CELLS, FIELDS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def score(pack, paths, *, expected_prediction_sha256=None, expected_provenance_sha256=None):
    pack = Path(pack)
    if len(paths) != 4 or len({Path(p).resolve() for p in paths}) != 4:
        raise ValueError('Four distinct review paths are required')
    meta_raw = (pack / 'private' / 'provenance.json').read_bytes()
    if expected_provenance_sha256 and sha(meta_raw) != expected_provenance_sha256:
        raise ValueError('Provenance hash mismatch')
    meta = json.loads(meta_raw); items = meta['items']
    if len(items) != 64 or set(x['cell'] for x in items.values()) != set(CELLS):
        raise ValueError('Unexpected review pack')
    for identifier, item in items.items():
        if Path(identifier).name != identifier:
            raise ValueError('Invalid sample identifier')
        if sha((pack / 'private' / 'images' / f'{identifier}.png').read_bytes()) != item['image_sha256']:
            raise ValueError('Image hash mismatch')
    reviews, hashes = [], []
    for path in map(Path, paths):
        hashes.append(sha(path.read_bytes()))
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != list(FIELDS):
                raise ValueError('Unexpected review columns')
            rows = list(reader)
        ids = [row['sample_id'] for row in rows]
        if len(ids) != len(items) or len(set(ids)) != len(ids) or set(ids) != set(items):
            raise ValueError('Missing, duplicate or unexpected IDs')
        reviews.append({r['sample_id']: {q: decision(r[q]) for q in FIELDS[1:]} for r in rows})
    per_image = {}
    for identifier in sorted(items):
        votes = {q: dict(Counter(r[identifier][q] for r in reviews)) for q in FIELDS[1:]}
        unanimous = {q: ('yes' if v.get('yes') == 4 else 'no' if v.get('no') == 4 else None)
                     for q, v in votes.items()}
        per_image[identifier] = {'votes': votes, 'unanimous': unanimous}
    cells = {}
    for cell in CELLS:
        cells[cell] = {}
        for color in ('red', 'blue'):
            subset = [key for key, item in items.items() if item['cell'] == cell and item['color'] == color]
            if len(subset) != 8:
                raise ValueError('Unexpected cell/color allocation')
            labels = [per_image[key]['unanimous']['target_circle'] for key in subset]
            cells[cell][color] = {'yes': labels.count('yes'), 'no': labels.count('no'),
                                  'unresolved': labels.count(None), 'total': len(labels)}
    # Worst-case bounds are descriptive bounds on these exact 16 images per cell;
    # disputed/unsure labels are never imputed as ground truth.
    bounds = {}
    for cell in CELLS:
        yes = sum(cells[cell][color]['yes'] for color in ('red', 'blue'))
        unknown = sum(cells[cell][color]['unresolved'] for color in ('red', 'blue'))
        bounds[cell] = [yes / 16, (yes + unknown) / 16]
    difference = [bounds['low-imbalanced'][0] - bounds['high-balanced'][1],
                  bounds['low-imbalanced'][1] - bounds['high-balanced'][0]]
    predictions_raw = (pack / 'private' / 'predictions.json').read_bytes()
    if expected_prediction_sha256 and sha(predictions_raw) != expected_prediction_sha256:
        raise ValueError('Prediction hash mismatch')
    predictions = json.loads(predictions_raw)
    if set(predictions) != set(items):
        raise ValueError('Prediction IDs mismatch')
    agreement = {}
    for checker in ('standard', 'variable_count'):
        counts = Counter()
        for identifier in items:
            truth = per_image[identifier]['unanimous']['target_circle']
            predicted = predictions[identifier][checker]
            if predicted is not None and type(predicted) is not bool:
                raise ValueError('Invalid checker prediction')
            if predicted is None:
                counts['abstained'] += 1
                counts['abstained_' + (truth if truth is not None else 'unresolved')] += 1
            elif truth is None:
                counts['review_unresolved'] += 1
            elif predicted == (truth == 'yes'):
                counts['agree'] += 1
                counts['tp' if predicted else 'tn'] += 1
            else:
                counts['disagree'] += 1
                counts['fp' if predicted else 'fn'] += 1
        agreement[checker] = dict(counts)
    return {'schema_version': 1, 'provenance_sha256': sha(meta_raw),
            'prediction_sha256': sha(predictions_raw), 'review_sha256': hashes,
            'per_image': per_image, 'cells_by_color': cells, 'cell_success_bounds': bounds,
            'difference_in_drops_bounds': difference, 'checker_comparison': agreement,
            'note': ('Bounds describe these images only, not population confidence intervals. '
                     'One paired training-seed block, unresolved labels and checker abstentions '
                     'prevent a Stage 0 or MOSAIC replication claim.')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--reviews', type=Path, nargs=4, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-predictions-sha256'); p.add_argument('--expected-provenance-sha256')
    args = p.parse_args()
    try:
        result = score(args.pack, args.reviews, expected_prediction_sha256=args.expected_predictions_sha256,
                       expected_provenance_sha256=args.expected_provenance_sha256)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        p.error(str(exc))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
