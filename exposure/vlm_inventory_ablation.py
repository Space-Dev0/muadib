"""Developmental object-list ablation that ignores the model's uncertainty flag.

This is a post-hoc analysis, not an independently validated checker.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from exposure import vlm_audit
from exposure.vlm_inventory_audit import COLORS, SHAPES


def decision(text: str, target: str) -> str:
    """Use explicit object attributes; malformed or unknown attributes abstain."""
    try:
        data = json.loads(text)
        objects = data['objects']
        if (type(data['uncertain']) is not bool or not isinstance(objects, list) or len(objects) > 8
                or any(not isinstance(obj, dict) or obj.get('color') not in COLORS
                       or obj.get('shape') not in SHAPES for obj in objects)):
            return 'unsure'
        if any(obj['color'] == target and obj['shape'] == 'circle' for obj in objects):
            return 'yes'
        if any(obj['color'] == 'unknown' or obj['shape'] == 'unknown' for obj in objects):
            return 'unsure'
        return 'no'
    except (ValueError, TypeError, KeyError):
        return 'unsure'


def score(pack: Path, output: Path) -> dict:
    from exposure.mosaic_matched_eval_score import score as score_reviews
    plan_raw = (output / 'plan.json').read_bytes(); plan = json.loads(plan_raw)
    model = json.loads((output / 'model.json').read_text())
    human = json.loads((pack / 'score.json').read_text())
    if (vlm_audit.sha(plan_raw) != model['plan_sha256']
            or human['provenance_sha256'] != plan['provenance_sha256']):
        raise ValueError('plan or review provenance mismatch')
    checked = score_reviews(pack, [pack / f'reviewer-{i}' / 'reviews.csv' for i in range(1, 5)],
                            expected_provenance_sha256=plan['provenance_sha256'],
                            expected_prediction_sha256=human['prediction_sha256'])
    if checked['review_sha256'] != human['review_sha256'] or checked['per_image'] != human['per_image']:
        raise ValueError('review CSVs do not match saved scores')
    counts = {truth: Counter() for truth in ('yes', 'no', 'unresolved')}
    records = []
    for item in plan['items']:
        raw = (output / 'results' / (item['id'] + '.json')).read_bytes()
        response = json.loads(raw)
        if response['item'] != item or response['model_digest'] != model['digest']:
            raise ValueError('model response differs from frozen plan')
        result = decision(response['inventory_response'], item['color'])
        truth = human['per_image'][item['id']]['unanimous']['target_circle']
        group = truth if truth in ('yes', 'no') else 'unresolved'
        counts[group][result] += 1
        records.append({'id': item['id'], 'decision': result, 'response_sha256': vlm_audit.sha(raw)})
    return {'model': model, 'counts': {t: {a: counts[t][a] for a in ('yes', 'no', 'unsure')}
                                      for t in counts}, 'records': records,
            'note': 'Post-hoc ablation ignores model uncertainty; not a validated replacement checker.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score(args.pack, args.inventory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
        print(json.dumps(result['counts'], indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
