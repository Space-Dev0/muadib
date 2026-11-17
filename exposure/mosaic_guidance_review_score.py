"""Summarize independent blinded guidance reviews without resolving disputes."""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

QUESTIONS = ('target_circle', 'exactly_two_objects')


def decision(value):
    """Accept an explicit answer optionally followed by a comma and a note."""
    answer, separator, note = value.partition(',')
    answer = answer.strip().lower()
    if answer in ('maybe', 'not sure'):
        answer = 'unsure'
    if answer not in ('yes', 'no', 'unsure') or (separator and not note.strip()):
        raise ValueError('Each response must be yes, no or unsure, optionally followed by a comma and note')
    return answer


def digest(data):
    return hashlib.sha256(data).hexdigest()


def score(pack, review_paths):
    pack = Path(pack)
    if len(review_paths) != 4 or len({Path(p).resolve() for p in review_paths}) != 4:
        raise ValueError('Four distinct review files are required')
    meta = json.loads((pack / 'private' / 'provenance.json').read_text())
    items = meta['items']
    for item, details in items.items():
        if Path(item).name != item:
            raise ValueError('Invalid sample identifier')
        if digest((pack / 'private' / 'images' / f'{item}.png').read_bytes()) != details['image_sha256']:
            raise ValueError(f'Private image hash mismatch: {item}')
    reviews = []
    hashes = []
    for path in map(Path, review_paths):
        raw = path.read_bytes()
        hashes.append(digest(raw))
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ['sample_id', *QUESTIONS]:
                raise ValueError(f'Unexpected columns: {path}')
            rows = list(reader)
        ids = [r['sample_id'] for r in rows]
        if len(ids) != len(items) or len(set(ids)) != len(ids) or set(ids) != set(items):
            raise ValueError(f'Missing, duplicate or unexpected sample ID: {path}')
        try:
            reviews.append({row['sample_id']: {q: decision(row[q]) for q in QUESTIONS} for row in rows})
        except (ValueError, AttributeError) as exc:
            raise ValueError(f'Invalid response in {path}: {exc}') from exc
    summaries = {}
    for question in QUESTIONS:
        counts = {item: dict(Counter(r[item][question] for r in reviews)) for item in sorted(items)}
        pairs = []
        for a, b in itertools.combinations(range(4), 2):
            comparable = [(reviews[a][item][question], reviews[b][item][question]) for item in items
                          if reviews[a][item][question] != 'unsure' and reviews[b][item][question] != 'unsure']
            pairs.append({'reviewers': [a + 1, b + 1], 'comparable': len(comparable),
                          'agree': sum(x == y for x, y in comparable)})
        summaries[question] = {'votes_by_image': counts, 'pairwise': pairs,
                               'unanimous_yes': sum(c.get('yes', 0) == 4 for c in counts.values()),
                               'unanimous_no': sum(c.get('no', 0) == 4 for c in counts.values()),
                               'any_unsure': sum(c.get('unsure', 0) > 0 for c in counts.values()),
                               'yes_no_disagreement': sum(c.get('yes', 0) > 0 and c.get('no', 0) > 0 for c in counts.values())}
    return {'schema_version': 1, 'image_count': len(items), 'checkpoint_sha256': meta['checkpoint_sha256'],
            'provenance_sha256': digest((pack / 'private' / 'provenance.json').read_bytes()),
            'review_sha256_in_argument_order': hashes, 'questions': summaries,
            'note': 'Review identities and independence cannot be verified from files; paired seeds and one model limit inference. No automatic adjudication.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--reviews', type=Path, nargs=4, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        result = score(args.pack, args.reviews)
        with args.output.open('x', encoding='utf-8') as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write('\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
