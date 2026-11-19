"""Run a blinded local Ollama vision-model audit against a frozen review pack.

The VLM is a candidate evaluator, not ground truth or a Stage 0 pass.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request
from urllib.parse import urlsplit

from PIL import Image

PROMPT = ("Look at the image carefully. Is there a clearly recognizable {color} circle? "
          "Answer yes only if a single object is both {color} and circular. "
          "Answer no if there is clearly no such object. Answer unsure if you cannot tell. "
          "Return only a JSON object with one field named answer: yes, no, or unsure.")
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string",
          "enum": ["yes", "no", "unsure"]}}, "required": ["answer"]}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(pack: Path, output: Path, *, seed: int = 82940,
            exclude: tuple[str, ...] = (), gpu_fit: bool = False) -> dict:
    """Freeze image order and integrity, without including human decisions."""
    raw = (pack / 'private' / 'provenance.json').read_bytes()
    provenance = json.loads(raw)
    if len(set(exclude)) != len(exclude) or set(exclude) - set(provenance['items']):
        raise ValueError('excluded IDs must be unique review images')
    items = []
    for identifier, meta in provenance['items'].items():
        if identifier in exclude:
            continue
        if Path(identifier).name != identifier or meta['color'] not in ('red', 'blue'):
            raise ValueError('invalid image identifier or color')
        data = (pack / 'private' / 'images' / (identifier + '.png')).read_bytes()
        if sha(data) != meta['image_sha256']:
            raise ValueError('review image hash mismatch')
        with Image.open(io.BytesIO(data)) as image:
            if image.mode != 'RGB' or image.size != (64, 64) or sha(image.tobytes()) != meta['pixel_sha256']:
                raise ValueError('invalid image pixels')
        items.append({'id': identifier, 'color': meta['color'], 'image_sha256': sha(data)})
    if not items:
        raise ValueError('no review images remain')
    random.Random(seed).shuffle(items)
    options = {'temperature': 0, 'seed': 42, 'num_predict': 64, 'num_ctx': 2048,
               'num_batch': 64}
    if gpu_fit:
        options.update({'num_ctx': 1024, 'num_batch': 16, 'num_gpu': 99})
    plan = {'schema': 1, 'provenance_sha256': sha(raw), 'seed': seed, 'excluded': list(exclude),
            'prompt': PROMPT, 'schema_format': SCHEMA, 'resize': 'nearest 64 to 256',
            'options': options,
            'items': items, 'code_sha256': sha(Path(__file__).read_bytes())}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'results').mkdir()
    with (output / 'plan.json').open('x', encoding='utf-8') as f:
        json.dump(plan, f, indent=2); f.write('\n')
    return plan


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError('invalid Ollama response')
    return result


def parse_answer(text: str) -> str:
    try:
        data = json.loads(text)
        answer = data.get('answer') if isinstance(data, dict) else None
        return answer if answer in ('yes', 'no', 'unsure') else 'unsure'
    except (ValueError, TypeError):
        return 'unsure'


def run(pack: Path, output: Path, *, url: str, model: str) -> None:
    """Resume fixed jobs; refuse a changed image, code, plan or model digest."""
    parsed_url = urlsplit(url)
    if (parsed_url.scheme != 'http' or parsed_url.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed_url.username or parsed_url.password or parsed_url.path not in ('', '/')):
        raise ValueError('only loopback Ollama URLs are supported')
    plan_raw = (output / 'plan.json').read_bytes()
    plan = json.loads(plan_raw)
    if (sha(Path(__file__).read_bytes()) != plan['code_sha256']
            or sha((pack / 'private' / 'provenance.json').read_bytes()) != plan['provenance_sha256']):
        raise ValueError('code or review provenance changed after freezing')
    tags = request_json(url.rstrip('/') + '/api/tags')['models']
    matches = [entry for entry in tags if entry.get('name') == model]
    if len(matches) != 1 or not matches[0].get('digest'):
        raise ValueError('model name missing or ambiguous in Ollama')
    model_info = {'model': model, 'digest': matches[0]['digest'], 'plan_sha256': sha(plan_raw)}
    model_path = output / 'model.json'
    if model_path.exists():
        if json.loads(model_path.read_text()) != model_info:
            raise ValueError('model or plan changed across runs')
    else:
        with model_path.open('x', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2); f.write('\n')
    for index, item in enumerate(plan['items'], 1):
        path = output / 'results' / (item['id'] + '.json')
        raw = (pack / 'private' / 'images' / (item['id'] + '.png')).read_bytes()
        if sha(raw) != item['image_sha256']:
            raise ValueError('review image changed after freezing')
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get('item') != item or existing.get('model_digest') != model_info['digest']:
                raise ValueError('existing response does not match frozen job')
            continue
        with Image.open(io.BytesIO(raw)) as image:
            buffer = io.BytesIO()
            image.resize((256, 256), Image.Resampling.NEAREST).save(buffer, 'PNG')
        payload = {'model': model, 'stream': False, 'format': SCHEMA,
                   'options': plan['options'],
                   'messages': [{'role': 'user', 'content': PROMPT.format(color=item['color']),
                                 'images': [base64.b64encode(buffer.getvalue()).decode()]}]}
        response = request_json(url.rstrip('/') + '/api/chat', payload)
        text = response['message']['content']
        record = {'item': item, 'model_digest': model_info['digest'], 'response': text,
                  'decision': parse_answer(text), 'done_reason': response.get('done_reason'),
                  'total_duration_ns': response.get('total_duration')}
        with path.open('x', encoding='utf-8') as f:
            json.dump(record, f, indent=2, allow_nan=False); f.write('\n')
        print(f'Completed {index}/{len(plan["items"])}: {item["id"]}', flush=True)


def score(pack: Path, output: Path) -> dict:
    """Compare frozen VLM decisions to saved human labels; count abstentions."""
    plan = json.loads((output / 'plan.json').read_text())
    model = json.loads((output / 'model.json').read_text())
    human = json.loads((pack / 'score.json').read_text())
    if (sha((output / 'plan.json').read_bytes()) != model['plan_sha256']
            or human['provenance_sha256'] != plan['provenance_sha256']):
        raise ValueError('plan or human score provenance mismatch')
    from exposure.mosaic_matched_eval_score import score as score_reviews
    verified = score_reviews(pack, [pack / f'reviewer-{i}' / 'reviews.csv' for i in range(1, 5)],
                             expected_provenance_sha256=plan['provenance_sha256'],
                             expected_prediction_sha256=human['prediction_sha256'])
    if (verified['review_sha256'] != human['review_sha256']
            or verified['per_image'] != human['per_image']):
        raise ValueError('saved human scores do not match review CSVs')
    counts = {truth: Counter() for truth in ('yes', 'no', 'unresolved')}
    for item in plan['items']:
        record = json.loads((output / 'results' / (item['id'] + '.json')).read_text())
        if record['item'] != item or record['model_digest'] != model['digest']:
            raise ValueError('response does not match frozen plan')
        answer = record['decision']
        if answer not in ('yes', 'no', 'unsure') or answer != parse_answer(record['response']):
            raise ValueError('invalid response decision')
        truth = human['per_image'][item['id']]['unanimous']['target_circle']
        counts[truth if truth in ('yes', 'no') else 'unresolved'][answer] += 1
    return {'model': model, 'counts': {k: {a: counts[k][a] for a in ('yes', 'no', 'unsure')}
                                      for k in counts},
            'note': 'Descriptive comparison with saved reviews; limited negatives and no independent model replicates.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'score'))
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:11435')
    parser.add_argument('--model', default='qwen2.5vl:3b')
    parser.add_argument('--exclude', action='append', default=[], help='Exclude a previously inspected image when preparing')
    parser.add_argument('--gpu-fit', action='store_true', help='Prepare smaller GPU-offload inference settings')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            print(json.dumps({'images': len(prepare(args.pack, args.output,
                                                   exclude=tuple(args.exclude), gpu_fit=args.gpu_fit)['items'])}))
        elif args.action == 'run':
            run(args.pack, args.output, url=args.url, model=args.model)
        else:
            result = score(args.pack, args.output)
            with (args.output / 'score.json').open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2); f.write('\n')
            print(json.dumps(result['counts'], indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
