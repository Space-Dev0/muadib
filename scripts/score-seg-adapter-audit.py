"""Score a pre-frozen experimental adapter against original pilot reviews.

Shared-image disagreements and unresolved answers remain unresolved. Do not
use this score to tune the same adapter and call it independent validation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from exposure.checker_seg_adapter import CONFIG
from exposure.human_audit import read_reviews


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(pack, predictions_path, expected_predictions, expected_code, expected_weights):
    if sha(predictions_path) != expected_predictions:
        raise ValueError('prediction SHA-256 mismatch')
    predictions = json.loads(predictions_path.read_text())
    provenance = json.loads((pack / 'private/provenance.json').read_text())
    manifest_path = pack / 'private/all-rules.json'
    manifest = json.loads(manifest_path.read_text())
    if predictions['manifest_sha256'] != sha(manifest_path):
        raise ValueError('manifest hash mismatch')
    if predictions['adapter_code_sha256'] != expected_code or predictions['checkpoint_sha256'] != expected_weights:
        raise ValueError('adapter or checkpoint hash mismatch')
    if predictions['config'] != CONFIG:
        raise ValueError('adapter config mismatch')
    ids = {s['sample_id'] for s in manifest['samples']}
    by_id = {}
    for p in predictions['predictions']:
        i = p['sample_id']
        if i in by_id or i not in ids or (p['decision'] is not None and type(p['decision']) is not bool):
            raise ValueError('duplicate or invalid prediction')
        if p['image_sha256'] != provenance['images'][i]['sha256']:
            raise ValueError('prediction image hash differs from provenance')
        if sha(pack / 'private/images' / (i + '.png')) != p['image_sha256']:
            raise ValueError('local image hash mismatch')
        by_id[i] = p['decision']
    if set(by_id) != ids:
        raise ValueError('predictions do not cover exact manifest')
    votes = defaultdict(list); review_hashes = {}
    for reviewer in ('aryan', 'kanishak', 'nikhil', 'vansh'):
        path = pack / f'{reviewer}-rule-reviews.csv'
        review_hashes[reviewer] = sha(path)
        for sample, answer in read_reviews(path).items():
            votes[sample].append(answer)
    if set(votes) != ids:
        raise ValueError('reviews do not cover exact manifest')
    shared = set(provenance['shared_ids'])
    if any(len(votes[i]) != (4 if i in shared else 1) for i in ids):
        raise ValueError('review assignment counts mismatch')
    result = Counter(); by_rule = defaultdict(Counter)
    samples = {s['sample_id']: s for s in manifest['samples']}
    for i in ids:
        values = votes[i]
        resolved = values[0] if values[0] is not None and all(v == values[0] for v in values) else None
        pred = by_id[i]
        if resolved is None:
            label = 'human_unresolved'
        elif pred is None:
            label = 'abstain_human_yes' if resolved else 'abstain_human_no'
        else:
            label = ('tp' if pred else 'fn') if resolved else ('fp' if pred else 'tn')
        result[label] += 1; by_rule[samples[i]['rule_name']][label] += 1
    return {'schema_version': 1, 'prediction_sha256': expected_predictions,
            'code_sha256': expected_code, 'weights_sha256': expected_weights,
            'manifest_sha256': sha(manifest_path), 'review_sha256': review_hashes,
            'total_images': len(ids), 'counts': dict(result),
            'by_rule': {k: dict(v) for k, v in by_rule.items()},
            'note': 'Exploratory on previously reviewed data. Accepted-only error fractions are conditional on selection; shared disagreements unresolved. No independent validation or Stage 0 pass.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, required=True); p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--expected-predictions', required=True); p.add_argument('--expected-code', required=True)
    p.add_argument('--expected-weights', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        result = score(a.pack, a.predictions, a.expected_predictions, a.expected_code, a.expected_weights)
        with a.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2); f.write('\n')
        print(json.dumps(result['counts']))
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))

if __name__ == '__main__': main()
