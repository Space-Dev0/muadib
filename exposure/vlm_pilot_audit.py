"""Evaluate frozen VLM inventories on the reviewed 800k single-attribute pilot.

This exploratory comparison covers red-object and square-object questions,
not relations or a representative generated-image population.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from exposure.audit_pilot_score import evaluate
from exposure.human_audit import read_reviews
from exposure.vlm_audit import sha
from exposure.vlm_inventory_audit import COLORS, SHAPES, prepare as prepare_inventory

REVIEWERS = {1: 'nikhil', 2: 'vansh', 3: 'aryan', 4: 'kanishak'}


def _review_paths(source: Path):
    return {index: (source / f'{name}-rule-reviews.csv', source / f'{name}-predicate-reviews.csv')
            for index, name in REVIEWERS.items()}


def _verify_source(source: Path) -> dict:
    # The returned archives were renamed by reviewers; restore their expected
    # names in a temporary view without modifying the originals.
    with tempfile.TemporaryDirectory(prefix='muadib-review-view-') as directory:
        view = Path(directory)
        (view / 'private').symlink_to((source / 'private').resolve(), target_is_directory=True)
        for index, name in REVIEWERS.items():
            (view / f'reviewer-{index}.zip').symlink_to((source / f'{name}.zip').resolve())
        return evaluate(view, _review_paths(source))


def prepare(source: Path, destination: Path, output: Path) -> dict:
    """Create a blind derivative pack and freeze its inventory prompt."""
    source = source.resolve(strict=True); destination = destination.resolve(); output = output.resolve()
    if (destination == output or destination.is_relative_to(source) or output.is_relative_to(source)
            or output.is_relative_to(destination) or destination.is_relative_to(output)):
        raise ValueError('source, derivative and output must be separate')
    _verify_source(source)
    private = source / 'private'
    provenance_raw = (private / 'provenance.json').read_bytes()
    rules_raw = (private / 'all-rules.json').read_bytes()
    provenance = json.loads(provenance_raw); rules = json.loads(rules_raw)['samples']
    selected = [row for row in rules if row['rule_name'] in ('red', 'square')]
    if Counter(row['rule_name'] for row in selected) != {'red': 75, 'square': 75}:
        raise ValueError('unexpected single-attribute inventory')
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'private'; (target / 'images').mkdir(parents=True)
    items, mapping = {}, {}
    for row in selected:
        identifier = row['sample_id']
        if Path(identifier).name != identifier:
            raise ValueError('invalid sample ID')
        raw = (private / 'images' / f'{identifier}.png').read_bytes()
        image = provenance['images'][identifier]
        if sha(raw) != image['sha256']:
            raise ValueError('image hash mismatch')
        (target / 'images' / f'{identifier}.png').write_bytes(raw)
        # Inventory inference never sends this placeholder target to the model.
        items[identifier] = {'color': 'red', 'image_sha256': sha(raw),
                             'pixel_sha256': image['pixel_sha256']}
        mapping[identifier] = row['rule_name']
    derived = {'items': items, 'source_provenance_sha256': sha(provenance_raw),
               'source_rules_sha256': sha(rules_raw), 'rule_map': mapping,
               'note': 'Red/square subset; no human labels or predictions supplied to the model.'}
    with (target / 'provenance.json').open('x', encoding='utf-8') as f:
        json.dump(derived, f, indent=2); f.write('\n')
    plan = prepare_inventory(destination, output)
    return {'images': len(plan['items']), 'rule_counts': dict(Counter(mapping.values()))}


def decide(text: str, rule: str) -> str:
    """Attribute-only version of the post-hoc inventory ablation."""
    try:
        obj = json.loads(text); items = obj['objects']
        if (type(obj['uncertain']) is not bool or not isinstance(items, list) or len(items) > 8
                or any(not isinstance(x, dict) or x.get('color') not in COLORS
                       or x.get('shape') not in SHAPES for x in items)):
            return 'unsure'
        field, value = ('color', 'red') if rule == 'red' else ('shape', 'square') if rule == 'square' else (None, None)
        if field is None:
            raise ValueError('unsupported rule')
        if any(x[field] == value for x in items):
            return 'yes'
        return 'unsure' if any(x[field] == 'unknown' for x in items) else 'no'
    except (TypeError, KeyError, json.JSONDecodeError):
        return 'unsure'


def score(source: Path, derivative: Path, output: Path, human_score: Path) -> dict:
    checked = _verify_source(source)
    plan_raw = (output / 'plan.json').read_bytes(); plan = json.loads(plan_raw)
    model = json.loads((output / 'model.json').read_text())
    derived_raw = (derivative / 'private' / 'provenance.json').read_bytes(); derived = json.loads(derived_raw)
    if (sha(plan_raw) != model['plan_sha256'] or sha(derived_raw) != plan['provenance_sha256']
            or derived['source_provenance_sha256'] != sha((source / 'private' / 'provenance.json').read_bytes())
            or derived['source_rules_sha256'] != sha((source / 'private' / 'all-rules.json').read_bytes())):
        raise ValueError('frozen provenance mismatch')
    saved = json.loads(human_score.read_text())
    if checked['review_sha256'] != saved['review_sha256']:
        raise ValueError('review files do not match the previously scored pilot')
    reviews = [read_reviews(pair[0]) for pair in _review_paths(source).values()]
    counts = {rule: {truth: Counter() for truth in ('yes', 'no', 'unresolved')} for rule in ('red', 'square')}
    records = []
    for item in plan['items']:
        identifier = item['id']; rule = derived['rule_map'][identifier]
        response_raw = (output / 'results' / f'{identifier}.json').read_bytes()
        response = json.loads(response_raw)
        if response['item'] != item or response['model_digest'] != model['digest']:
            raise ValueError('model response differs from frozen job')
        votes = [review[identifier] for review in reviews if identifier in review]
        if not votes:
            raise ValueError('missing human vote')
        truth = ('yes' if votes[0] else 'no') if len(set(votes)) == 1 else 'unresolved'
        prediction = decide(response['inventory_response'], rule)
        counts[rule][truth][prediction] += 1
        records.append({'id': identifier, 'rule': rule, 'truth': truth, 'human_votes': len(votes),
                        'decision': prediction, 'response_sha256': sha(response_raw)})
    return {'model': model, 'counts': {r: {t: {v: counts[r][t][v] for v in ('yes', 'no', 'unsure')}
                                            for t in counts[r]} for r in counts}, 'records': records,
            'note': 'Prior reviewed 800k single-attribute sample; mostly one reviewer per image; post-hoc model ablation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'score'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--derivative', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--human-score', type=Path, help='Previously scored pilot report (required for scoring)')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            print(json.dumps(prepare(args.source, args.derivative, args.output), indent=2))
        else:
            if args.human_score is None:
                raise ValueError('scoring requires --human-score')
            result = score(args.source, args.derivative, args.output, args.human_score)
            with (args.output / 'pilot-score.json').open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2); f.write('\n')
            print(json.dumps(result['counts'], indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
