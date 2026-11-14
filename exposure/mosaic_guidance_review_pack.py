"""Prepare a blinded, limited human readiness review of paired guidance samples."""
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


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(source: Path, output: Path, *, seed: int = 81386):
    """Freeze the eight recorded scale-1/2 images without exposing scale or seed."""
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output must be separate')
    report_path = source / 'report.json'
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    expected = {(color, sample_seed, scale) for color in ('red', 'blue')
                for sample_seed in (81380, 81381) for scale in (1.0, 2.0)}
    rows = report['images']
    keys = [(r['color'], r['seed'], r['guidance']) for r in rows]
    if len(keys) != 8 or set(keys) != expected:
        raise ValueError('Expected exactly the frozen eight guidance samples')
    items = []
    for row in rows:
        filename = row['file']
        if Path(filename).name != filename:
            raise ValueError('Invalid source filename')
        raw = (source / filename).read_bytes()
        if digest(raw) != row['sha256']:
            raise ValueError(f'Image hash mismatch: {filename}')
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.format != 'PNG' or image.mode != 'RGB' or image.size != (64, 64):
                raise ValueError(f'Invalid image: {filename}')
            pixels = image.tobytes()
            if digest(pixels) != row['pixel_sha256']:
                raise ValueError(f'Pixel hash mismatch: {filename}')
            clean = Image.frombytes('RGB', image.size, pixels)
            buf = io.BytesIO()
            clean.save(buf, format='PNG')
        items.append({'color': row['color'], 'seed': row['seed'], 'guidance': row['guidance'],
                      'png': buf.getvalue(), 'pixel_sha256': digest(pixels)})
    rng = random.Random(seed)
    rng.shuffle(items)
    for i, item in enumerate(items, 1):
        item['id'] = f'readiness-{i:03d}'
    output.mkdir(parents=True, exist_ok=False)
    private = output / 'private'
    private.mkdir(mode=0o700)
    images = private / 'images'
    images.mkdir()
    for item in items:
        (images / (item['id'] + '.png')).write_bytes(item['png'])
    provenance = {'seed': seed, 'source_report_sha256': digest(report_bytes),
                  'packager_sha256': digest(Path(__file__).read_bytes()),
                  'checkpoint_sha256': report['checkpoint_sha256'],
                  'items': {item['id']: {'color': item['color'], 'seed': item['seed'],
                             'guidance': item['guidance'], 'pixel_sha256': item['pixel_sha256'],
                             'image_sha256': digest(item['png'])} for item in items},
                  'note': 'Post-hoc readiness check, one model and two paired seeds; not a Stage 0 audit.'}
    (private / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    table = io.StringIO(newline='')
    writer = csv.writer(table, lineterminator='\n')
    writer.writerow(['sample_id', 'target_circle', 'exactly_two_objects'])
    writer.writerows((item['id'], '', '') for item in items)
    page = ('<!doctype html><html><meta charset="utf-8"><title>Shape readiness review</title>'
            '<style>body{font:16px sans-serif;max-width:950px;margin:auto}td,th{padding:8px;'
            'border-bottom:1px solid #ccc;text-align:left}img{image-rendering:pixelated}</style>'
            '<h1>Shape readiness review</h1><p>Work independently. For each image enter yes, no, '
            'or unsure in each column of reviews.csv. Count distinct recognizable objects, not '
            'background speckles. Use unsure whenever a shape or count cannot be determined. '
            'Do not discuss responses until everyone finishes.</p>'
            '<table><tr><th>ID</th><th>Image</th><th>target_circle</th><th>exactly_two_objects</th></tr>')
    for item in items:
        identifier, color = html.escape(item['id']), html.escape(item['color'])
        page += (f'<tr><td>{identifier}</td><td><img src="images/{identifier}.png" width="192" height="192"></td>'
                 f'<td>Is there a {color} circular object?</td><td>Are exactly two distinct objects visible?</td></tr>')
    page += '</table></html>'
    hashes = {}
    for reviewer in range(1, 5):
        path = output / f'reviewer-{reviewer}.zip'
        with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('review.html', page)
            archive.writestr('reviews.csv', table.getvalue())
            for item in items:
                archive.writestr(f"images/{item['id']}.png", item['png'])
        hashes[path.name] = digest(path.read_bytes())
    (private / 'archive-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return {'images': len(items), 'reviewers': 4, 'note': provenance['note']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(build(a.source, a.output), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
