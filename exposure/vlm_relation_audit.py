"""Blind local-VLM comparison on the reviewed 800k spatial-relation subset.

The model answers rules directly. This is exploratory checker evaluation, not
ground truth or a Stage 0 pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import base64
import io
import json
from pathlib import Path
import random
from urllib.parse import urlsplit

from PIL import Image
from exposure import vlm_audit
from exposure.human_audit import read_reviews
from exposure.vlm_pilot_audit import _review_paths, _verify_source

PROMPTS = {
    'red_left_of_square': 'Are there two distinct objects where one is red and its center is strictly left of the center of a square?',
    'blue_above_circle': 'Are there two distinct objects where one is blue and its center is strictly above the center of a circle?',
}
SUFFIX = ' Answer yes if the relation is clearly true, no if clearly false, or unsure if you cannot tell. Return only JSON with an answer field.'
OPTIONS = {'temperature': 0, 'seed': 42, 'num_predict': 64, 'num_ctx': 1024, 'num_batch': 16, 'num_gpu': 99}


def prepare(source: Path, output: Path, *, seed: int = 82960, low_memory: bool = False) -> dict:
    source = source.resolve(strict=True); output = output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('source and output must be separate')
    _verify_source(source)
    provenance_raw = (source / 'private' / 'provenance.json').read_bytes()
    rules_raw = (source / 'private' / 'all-rules.json').read_bytes()
    provenance = json.loads(provenance_raw)
    selected = [row for row in json.loads(rules_raw)['samples'] if row['rule_name'] in PROMPTS]
    if Counter(row['rule_name'] for row in selected) != {name: 75 for name in PROMPTS}:
        raise ValueError('unexpected relation inventory')
    items = []
    for row in selected:
        identifier = row['sample_id']
        if Path(identifier).name != identifier:
            raise ValueError('invalid image ID')
        raw = (source / 'private' / 'images' / f'{identifier}.png').read_bytes()
        if vlm_audit.sha(raw) != provenance['images'][identifier]['sha256']:
            raise ValueError('image hash mismatch')
        items.append({'id': identifier, 'rule': row['rule_name'], 'image_sha256': vlm_audit.sha(raw)})
    if len({item['id'] for item in items}) != len(items):
        raise ValueError('duplicate image IDs')
    random.Random(seed).shuffle(items)
    options = dict(OPTIONS)
    if low_memory:
        options.update({'num_ctx': 512, 'num_batch': 1, 'num_thread': 1})
    plan = {'schema': 1, 'source_provenance_sha256': vlm_audit.sha(provenance_raw),
            'source_rules_sha256': vlm_audit.sha(rules_raw), 'seed': seed,
            'prompts': PROMPTS, 'suffix': SUFFIX, 'schema_format': vlm_audit.SCHEMA,
            'options': options, 'resize': 'nearest 64 to 256', 'items': items,
            'code_sha256': vlm_audit.sha(Path(__file__).read_bytes()),
            'helper_sha256': vlm_audit.sha(Path(vlm_audit.__file__).read_bytes())}
    output.mkdir(parents=True, exist_ok=False); (output / 'results').mkdir()
    with (output / 'plan.json').open('x', encoding='utf-8') as f:
        json.dump(plan, f, indent=2); f.write('\n')
    return plan


def run(source: Path, output: Path, *, url: str, model: str) -> None:
    endpoint = urlsplit(url)
    if (endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1')
            or endpoint.username or endpoint.password or endpoint.path not in ('', '/')):
        raise ValueError('only loopback Ollama URLs are supported')
    plan_raw = (output / 'plan.json').read_bytes(); plan = json.loads(plan_raw)
    if (vlm_audit.sha(Path(__file__).read_bytes()) != plan['code_sha256']
            or vlm_audit.sha(Path(vlm_audit.__file__).read_bytes()) != plan['helper_sha256']
            or vlm_audit.sha((source / 'private' / 'provenance.json').read_bytes()) != plan['source_provenance_sha256']
            or vlm_audit.sha((source / 'private' / 'all-rules.json').read_bytes()) != plan['source_rules_sha256']):
        raise ValueError('code or source changed after freezing')
    matches = [entry for entry in vlm_audit.request_json(url.rstrip('/') + '/api/tags')['models']
               if entry.get('name') == model]
    if len(matches) != 1 or not matches[0].get('digest'):
        raise ValueError('model missing or ambiguous')
    info = {'model': model, 'digest': matches[0]['digest'], 'plan_sha256': vlm_audit.sha(plan_raw)}
    model_path = output / 'model.json'
    if model_path.exists():
        if json.loads(model_path.read_text()) != info:
            raise ValueError('model or plan changed')
    else:
        with model_path.open('x', encoding='utf-8') as f:
            json.dump(info, f, indent=2); f.write('\n')
    for index, item in enumerate(plan['items'], 1):
        path = output / 'results' / f"{item['id']}.json"
        raw = (source / 'private' / 'images' / f"{item['id']}.png").read_bytes()
        if vlm_audit.sha(raw) != item['image_sha256']:
            raise ValueError('image changed')
        if path.exists():
            result = json.loads(path.read_text())
            if (result['item'] != item or result['model_digest'] != info['digest']
                    or result['decision'] != vlm_audit.parse_answer(result['response'])):
                raise ValueError('saved response mismatch')
            continue
        with Image.open(io.BytesIO(raw)) as image:
            buffer = io.BytesIO(); image.resize((256, 256), Image.Resampling.NEAREST).save(buffer, 'PNG')
        payload = {'model': model, 'stream': False, 'format': plan['schema_format'], 'options': plan['options'],
                   'messages': [{'role': 'user', 'content': plan['prompts'][item['rule']] + plan['suffix'],
                                 'images': [base64.b64encode(buffer.getvalue()).decode()]}]}
        response = vlm_audit.request_json(url.rstrip('/') + '/api/chat', payload)
        text = response['message']['content']
        record = {'item': item, 'model_digest': info['digest'], 'response': text,
                  'decision': vlm_audit.parse_answer(text), 'done_reason': response.get('done_reason')}
        with path.open('x', encoding='utf-8') as f:
            json.dump(record, f, indent=2); f.write('\n')
        print(f'Completed {index}/{len(plan["items"])}: {item["id"]}', flush=True)


def score(source: Path, output: Path, human_score: Path) -> dict:
    checked = _verify_source(source); saved = json.loads(human_score.read_text())
    if checked['review_sha256'] != saved['review_sha256']:
        raise ValueError('review files differ from previously scored pilot')
    plan_raw = (output / 'plan.json').read_bytes(); plan = json.loads(plan_raw)
    info = json.loads((output / 'model.json').read_text())
    if (vlm_audit.sha(plan_raw) != info['plan_sha256']
            or vlm_audit.sha(Path(__file__).read_bytes()) != plan['code_sha256']
            or vlm_audit.sha(Path(vlm_audit.__file__).read_bytes()) != plan['helper_sha256']
            or vlm_audit.sha((source / 'private' / 'provenance.json').read_bytes()) != plan['source_provenance_sha256']
            or vlm_audit.sha((source / 'private' / 'all-rules.json').read_bytes()) != plan['source_rules_sha256']):
        raise ValueError('source or plan mismatch')
    reviews = [read_reviews(pair[0]) for pair in _review_paths(source).values()]
    counts = {rule: {truth: Counter() for truth in ('yes', 'no', 'unresolved')} for rule in PROMPTS}
    records = []
    for item in plan['items']:
        raw = (output / 'results' / f"{item['id']}.json").read_bytes(); response = json.loads(raw)
        if (response['item'] != item or response['model_digest'] != info['digest']
                or response['decision'] != vlm_audit.parse_answer(response['response'])):
            raise ValueError('response differs from frozen job')
        votes = [review[item['id']] for review in reviews if item['id'] in review]
        if not votes:
            raise ValueError('missing human vote')
        truth = ('yes' if votes[0] else 'no') if len(set(votes)) == 1 else 'unresolved'
        counts[item['rule']][truth][response['decision']] += 1
        records.append({'id': item['id'], 'rule': item['rule'], 'truth': truth, 'human_votes': len(votes),
                        'decision': response['decision'], 'response_sha256': vlm_audit.sha(raw)})
    return {'model': info, 'counts': {r: {t: {d: counts[r][t][d] for d in ('yes', 'no', 'unsure')}
                                         for t in counts[r]} for r in counts}, 'records': records,
            'note': 'Previously reviewed relation subset; mostly one reviewer per image; exploratory.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'score'))
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:11435'); parser.add_argument('--model', default='qwen2.5vl:3b')
    parser.add_argument('--human-score', type=Path)
    parser.add_argument('--low-memory', action='store_true', help='Freeze smaller batch/context and one CPU thread')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            print(json.dumps({'images': len(prepare(args.source, args.output,
                                                   low_memory=args.low_memory)['items'])}))
        elif args.action == 'run':
            run(args.source, args.output, url=args.url, model=args.model)
        else:
            if args.human_score is None: raise ValueError('scoring requires --human-score')
            result = score(args.source, args.output, args.human_score)
            with (args.output / 'score.json').open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2); f.write('\n')
            print(json.dumps(result['counts'], indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
