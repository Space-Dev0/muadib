"""Evaluate a local VLM object inventory against frozen reviewed images.

The model does not receive the target color; code evaluates the requested
circle from its listed objects. This is an exploratory checker candidate.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image
from exposure import vlm_audit

COLORS = ('red', 'blue', 'green', 'yellow', 'other', 'unknown')
SHAPES = ('circle', 'square', 'triangle', 'other', 'unknown')
PROMPT = ("List each clearly visible colored geometric object in this image. "
          "For each object give its color and shape. Ignore the background and tiny artifacts. "
          "Set uncertain to true if any object is too unclear to identify. Return only JSON.")
SCHEMA = {'type': 'object', 'properties': {
    'objects': {'type': 'array', 'maxItems': 8, 'items': {'type': 'object', 'properties': {
        'color': {'type': 'string', 'enum': list(COLORS)},
        'shape': {'type': 'string', 'enum': list(SHAPES)}}, 'required': ['color', 'shape']}},
    'uncertain': {'type': 'boolean'}}, 'required': ['objects', 'uncertain']}


def decide(text: str, target: str) -> str:
    try:
        data = json.loads(text)
        objects = data['objects']
        if (type(data['uncertain']) is not bool or not isinstance(objects, list) or len(objects) > 8
                or any(not isinstance(obj, dict) or obj.get('color') not in COLORS
                       or obj.get('shape') not in SHAPES for obj in objects)):
            return 'unsure'
        if any(obj['color'] == target and obj['shape'] == 'circle' for obj in objects):
            return 'yes'
        if data['uncertain'] or any(obj['color'] == 'unknown' or obj['shape'] == 'unknown'
                                    for obj in objects):
            return 'unsure'
        return 'no'
    except (ValueError, TypeError, KeyError):
        return 'unsure'


def prepare(pack: Path, output: Path, *, exclude: tuple[str, ...] = ()) -> dict:
    plan = vlm_audit.prepare(pack, output, exclude=exclude, gpu_fit=True)
    plan.update({'inventory_prompt': PROMPT, 'inventory_schema': SCHEMA,
                 'inventory_options': {**plan['options'], 'num_predict': 256},
                 'inventory_code_sha256': vlm_audit.sha(Path(__file__).read_bytes())})
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    return plan


def run(pack: Path, output: Path, *, url: str, model: str) -> None:
    endpoint = urlsplit(url)
    if (endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1')
            or endpoint.username or endpoint.password or endpoint.path not in ('', '/')):
        raise ValueError('only loopback Ollama URLs are supported')
    raw_plan = (output / 'plan.json').read_bytes(); plan = json.loads(raw_plan)
    if (vlm_audit.sha(Path(__file__).read_bytes()) != plan['inventory_code_sha256']
            or vlm_audit.sha(Path(vlm_audit.__file__).read_bytes()) != plan['code_sha256']
            or vlm_audit.sha((pack / 'private' / 'provenance.json').read_bytes()) != plan['provenance_sha256']):
        raise ValueError('code or review provenance changed after freezing')
    tags = vlm_audit.request_json(url.rstrip('/') + '/api/tags')['models']
    matches = [entry for entry in tags if entry.get('name') == model]
    if len(matches) != 1 or not matches[0].get('digest'):
        raise ValueError('model name missing or ambiguous')
    info = {'model': model, 'digest': matches[0]['digest'], 'plan_sha256': vlm_audit.sha(raw_plan)}
    model_path = output / 'model.json'
    if model_path.exists():
        if json.loads(model_path.read_text()) != info:
            raise ValueError('model or plan changed across runs')
    else:
        with model_path.open('x', encoding='utf-8') as f:
            json.dump(info, f, indent=2); f.write('\n')
    for index, item in enumerate(plan['items'], 1):
        path = output / 'results' / (item['id'] + '.json')
        raw = (pack / 'private' / 'images' / (item['id'] + '.png')).read_bytes()
        if vlm_audit.sha(raw) != item['image_sha256']:
            raise ValueError('review image changed after freezing')
        if path.exists():
            existing = json.loads(path.read_text())
            if (existing.get('item') != item or existing.get('model_digest') != info['digest']
                    or existing.get('decision') != decide(existing.get('inventory_response'), item['color'])):
                raise ValueError('existing inventory does not match frozen job')
            continue
        with Image.open(io.BytesIO(raw)) as image:
            buffer = io.BytesIO(); image.resize((256, 256), Image.Resampling.NEAREST).save(buffer, 'PNG')
        payload = {'model': model, 'stream': False, 'format': SCHEMA, 'options': plan['inventory_options'],
                   'messages': [{'role': 'user', 'content': PROMPT,
                                 'images': [base64.b64encode(buffer.getvalue()).decode()]}]}
        response = vlm_audit.request_json(url.rstrip('/') + '/api/chat', payload)
        text = response['message']['content']; decision = decide(text, item['color'])
        record = {'item': item, 'model_digest': info['digest'], 'inventory_response': text,
                  'response': json.dumps({'answer': decision}), 'decision': decision,
                  'done_reason': response.get('done_reason'), 'total_duration_ns': response.get('total_duration')}
        with path.open('x', encoding='utf-8') as f:
            json.dump(record, f, indent=2, allow_nan=False); f.write('\n')
        print(f'Completed {index}/{len(plan["items"])}: {item["id"]}', flush=True)


def score(pack: Path, output: Path) -> dict:
    plan = json.loads((output / 'plan.json').read_text())
    for item in plan['items']:
        record = json.loads((output / 'results' / (item['id'] + '.json')).read_text())
        if record['decision'] != decide(record['inventory_response'], item['color']):
            raise ValueError('inventory decision does not match recorded model output')
    return vlm_audit.score(pack, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'score'))
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:11435')
    parser.add_argument('--model', default='qwen2.5vl:3b')
    parser.add_argument('--exclude', action='append', default=[])
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            print(json.dumps({'images': len(prepare(args.pack, args.output, exclude=tuple(args.exclude))['items'])}))
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
