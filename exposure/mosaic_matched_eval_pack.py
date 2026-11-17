"""Freeze checker outputs and prepare blinded review packs for the 500k evaluation."""
import argparse
import csv
import hashlib
import html
import io
import json
import random
import zipfile
from pathlib import Path

from PIL import Image
from exposure import checker
from exposure.checker import DetectionError, detect, evaluate_variable_count_experimental
from exposure.rules import Attribute, satisfies

CELLS = ('low-balanced', 'low-imbalanced', 'high-balanced', 'high-imbalanced')
SEEDS = list(range(82000, 82008))
FIELDS = ('sample_id', 'target_circle', 'exactly_two_objects', 'exactly_three_objects')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(source, output, *, seed=82020, seeds=None):
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output must be separate')
    if not (source / 'COMPLETE').is_file():
        raise ValueError('Sampling did not complete')
    protocol_raw = (source / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_raw)
    expected_seeds = SEEDS if seeds is None else list(seeds)
    if expected_seeds not in (SEEDS, list(range(83000, 83008))):
        raise ValueError('Unsupported prespecified seed block')
    if protocol['seeds'] != expected_seeds or set(protocol['cells']) != set(CELLS) or protocol['guidance'] != 1.0:
        raise ValueError('Unexpected protocol')
    items, reports = [], {}
    for cell in CELLS:
        directory = source / cell
        report_raw = (directory / 'report.json').read_bytes()
        report = json.loads(report_raw)
        if (report['checkpoint_sha256'] != protocol['cells'][cell]['sha256'] or
                report['seeds'] != expected_seeds or report['guidance'] != [1.0]):
            raise ValueError(f'Checkpoint or sampling mismatch: {cell}')
        rows = report['images']
        keys = [(r['color'], r['seed'], r['guidance']) for r in rows]
        expected = {(color, s, 1.0) for color in ('red', 'blue') for s in expected_seeds}
        if len(keys) != 16 or set(keys) != expected:
            raise ValueError(f'Unexpected image set: {cell}')
        reports[cell] = sha(report_raw)
        for row in rows:
            if Path(row['file']).name != row['file']:
                raise ValueError('Invalid image path')
            raw = (directory / row['file']).read_bytes()
            if sha(raw) != row['sha256']:
                raise ValueError('Image hash mismatch')
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if image.format != 'PNG' or image.mode != 'RGB' or image.size != (64, 64):
                    raise ValueError('Invalid image')
                pixels = image.tobytes()
                if sha(pixels) != row['pixel_sha256']:
                    raise ValueError('Pixel hash mismatch')
                out = io.BytesIO()
                Image.frombytes('RGB', image.size, pixels).save(out, format='PNG')
            items.append({'cell': cell, 'color': row['color'], 'seed': row['seed'],
                          'png': out.getvalue(), 'pixel_sha256': sha(pixels), 'source_sha256': sha(raw)})
    random.Random(seed).shuffle(items)
    for index, item in enumerate(items, 1):
        item['id'] = f'matched-{index:04d}'
    output.mkdir(parents=True, exist_ok=False)
    private = output / 'private'; private.mkdir(mode=0o700)
    images = private / 'images'; images.mkdir()
    predictions = {}
    for item in items:
        identifier = item['id']
        (images / f'{identifier}.png').write_bytes(item['png'])
        rule = (Attribute('a', 'color', item['color']), Attribute('a', 'shape', 'circle'))
        decisions = {}
        with Image.open(io.BytesIO(item['png'])) as image:
            for name in ('standard', 'variable_count'):
                try:
                    decisions[name] = (satisfies(detect(image), rule) if name == 'standard'
                                       else evaluate_variable_count_experimental(image, rule))
                except DetectionError:
                    decisions[name] = None
        predictions[identifier] = decisions
    provenance = {'schema_version': 1, 'assignment_seed': seed, 'protocol_sha256': sha(protocol_raw),
                  'report_sha256': reports, 'checker_sha256': sha(Path(checker.__file__).read_bytes()),
                  'packager_sha256': sha(Path(__file__).read_bytes()),
                  'items': {x['id']: {k: x[k] for k in ('cell', 'color', 'seed', 'pixel_sha256', 'source_sha256')}
                            | {'image_sha256': sha(x['png'])} for x in items},
                  'note': '64 images, one matched model-seed block; not a Stage 0 gate.'}
    for name, value in (('provenance.json', provenance), ('predictions.json', predictions)):
        with (private / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    table = io.StringIO(newline=''); writer = csv.writer(table, lineterminator='\n')
    writer.writerow(FIELDS); writer.writerows((x['id'], '', '', '') for x in items)
    page = ('<!doctype html><html><meta charset="utf-8"><title>Matched model review</title>'
            '<style>body{font:16px sans-serif;max-width:950px;margin:auto}td,th{padding:8px;'
            'border-bottom:1px solid #ccc;text-align:left}img{image-rendering:pixelated}</style>'
            '<h1>Matched model review</h1><p>Work independently. Enter yes, no or unsure in '
            'reviews.csv for each question. Count distinct recognizable objects, not background '
            'speckles; use unsure when the answer cannot be determined. Do not discuss until all '
            'reviewers finish.</p><table><tr><th>ID</th><th>Image</th><th>Target circle</th>'
            '<th>Exactly two objects</th><th>Exactly three objects</th></tr>')
    for item in items:
        identifier = html.escape(item['id']); color = html.escape(item['color'])
        page += (f'<tr><td>{identifier}</td><td><img src="images/{identifier}.png" width="192" height="192"></td>'
                 f'<td>Is there a {color} circle?</td><td>Exactly two objects?</td><td>Exactly three objects?</td></tr>')
    page += '</table></html>'
    hashes = {}
    for reviewer in range(1, 5):
        path = output / f'reviewer-{reviewer}.zip'
        with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('review.html', page); archive.writestr('reviews.csv', table.getvalue())
            for item in items:
                archive.writestr(f"images/{item['id']}.png", item['png'])
        hashes[path.name] = sha(path.read_bytes())
    (private / 'archive-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    return {'images': len(items), 'reviewers': 4, 'note': provenance['note']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--second-block', action='store_true')
    args = p.parse_args()
    try:
        print(json.dumps(build(args.source, args.output,
                               seeds=range(83000, 83008) if args.second_block else None), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
