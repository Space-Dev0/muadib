"""Blinded review packs for the exploratory one-block multiscale 2x2 previews."""
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

from exposure import checker as checker_module
from exposure.checker import DetectionError, detect, evaluate_variable_count_experimental
from exposure.rules import Attribute, satisfies

CELLS = (('low', 'balanced'), ('low', 'imbalanced'), ('high', 'balanced'), ('high', 'imbalanced'))
FIELDS = ('sample_id', 'target_circle', 'exactly_two_objects', 'exactly_three_objects')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(source: Path, output: Path, *, seed: int = 81340):
    """Read all 16 fixed 20k previews, freeze private decisions and blind cell identities."""
    if type(seed) is not int or seed < 0:
        raise ValueError('Assignment seed must be nonnegative')
    source = source.resolve(strict=True)
    if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
        raise ValueError('Source and output must be separate')
    items, common, reports = [], None, {}
    for complexity, imbalance in CELLS:
        cell = f'{complexity}-{imbalance}'
        directory = source / cell
        report_path = directory / 'report.json'
        report_raw = report_path.read_bytes()
        report = json.loads(report_raw)
        keys = ('training_seed', 'width', 'batch_size', 'learning_rate', 'device', 'torch_version',
                'dataset_manifest_sha256', 'sample_seeds', 'sampler_steps')
        config = tuple(json.dumps(report[key], sort_keys=True) for key in keys)
        if report.get('cell') != [complexity, imbalance] or (common is not None and config != common):
            raise ValueError(f'Cell or matched configuration mismatch: {cell}')
        if report['sample_seeds'] != [81330, 81331] or report['sampler_steps'] != 50:
            raise ValueError('Unexpected frozen sampler configuration')
        snapshot = next((x for x in report['snapshots'] if x.get('step') == 20000), None)
        if snapshot is None:
            raise ValueError(f'Missing 20k snapshot: {cell}')
        common = config
        reports[cell] = {'sha256': sha(report_raw), 'snapshot_sha256': snapshot['checkpoint_sha256'],
                         'source_sha256': report['source_sha256']}
        for label in ('red', 'blue'):
            for generation_seed in (81330, 81331):
                path = directory / f'20000-{label}-{generation_seed}.png'
                raw = path.read_bytes()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.format != 'PNG' or image.mode != 'RGB' or image.size != (64, 64):
                        raise ValueError(f'Invalid image: {path}')
                    clean = Image.frombytes('RGB', image.size, image.tobytes())
                    encoded = io.BytesIO()
                    clean.save(encoded, format='PNG')
                    data = encoded.getvalue()
                items.append({'cell': cell, 'label': label, 'generation_seed': generation_seed,
                              'source_sha256': sha(raw), 'png': data, 'pixel_sha256': sha(clean.tobytes())})
    rng = random.Random(seed)
    rng.shuffle(items)
    for index, item in enumerate(items, 1):
        item['id'] = f'mosaic-{index:04d}'
    output.mkdir(parents=True, exist_ok=False)
    private = output / 'private'
    private.mkdir(mode=0o700)
    images = private / 'images'
    images.mkdir()
    predictions = {}
    for item in items:
        identifier = item['id']
        (images / f'{identifier}.png').write_bytes(item['png'])
        predicates = (Attribute('a', 'color', item['label']), Attribute('a', 'shape', 'circle'))
        values = {}
        with Image.open(io.BytesIO(item['png'])) as image:
            for name in ('standard', 'variable_count'):
                try:
                    values[name] = satisfies(detect(image), predicates) if name == 'standard' else evaluate_variable_count_experimental(image, predicates)
                except DetectionError:
                    values[name] = None
        predictions[identifier] = values
    provenance = {'schema_version': 1, 'seed': seed, 'reports': reports,
                  'checker_source_sha256': sha(Path(checker_module.__file__).read_bytes()),
                  'items': {x['id']: {k: x[k] for k in ('cell', 'label', 'generation_seed', 'source_sha256', 'pixel_sha256')}
                            | {'image_sha256': sha(x['png'])} for x in items},
                  'note': '16 previews, one training seed block; source hashes differ for the first cell. Not a gate or accuracy estimate.'}
    (private / 'predictions.json').write_text(json.dumps(predictions, indent=2) + '\n')
    (private / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    text = io.StringIO(newline='')
    writer = csv.writer(text, lineterminator='\n')
    writer.writerow(FIELDS)
    writer.writerows((x['id'], '', '', '') for x in items)
    page = ('<!doctype html><html><meta charset="utf-8"><title>Exploratory shapes review</title>'
            '<style>body{font:16px sans-serif;max-width:950px;margin:auto}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}img{image-rendering:pixelated}</style>'
            '<h1>Exploratory shapes review</h1><p>Work independently and do not discuss until all reviewers finish. '
            'For every image, complete each column in reviews.csv with yes, no, or unsure. '
            'Count distinct recognizable objects, not background speckles; use unsure when you cannot tell. '
            'Use the requested color shown for the circle question. The two count questions are separate.</p>'
            '<table><tr><th>ID</th><th>Image</th><th>target_circle</th><th>exactly_two_objects</th><th>exactly_three_objects</th></tr>')
    for item in items:
        identifier, label = item['id'], item['label']
        page += (f'<tr><td>{html.escape(identifier)}</td><td><img src="images/{identifier}.png" width="192" height="192"></td>'
                 f'<td>Is there a {html.escape(label)} circular object?</td><td>Are exactly two distinct objects visible?</td>'
                 '<td>Are exactly three distinct objects visible?</td></tr>')
    page += '</table></html>'
    archives = {}
    for reviewer in range(1, 5):
        archive_path = output / f'reviewer-{reviewer}.zip'
        with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('review.html', page)
            archive.writestr('reviews.csv', text.getvalue())
            for item in items:
                archive.writestr(f"images/{item['id']}.png", item['png'])
        archives[archive_path.name] = sha(archive_path.read_bytes())
    (private / 'archive-hashes.json').write_text(json.dumps(archives, indent=2) + '\n')
    return {'images': len(items), 'reviews_per_person': len(items),
            'standard_accepted': sum(x['standard'] is not None for x in predictions.values()),
            'variable_count_accepted': sum(x['variable_count'] is not None for x in predictions.values()),
            'note': provenance['note']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
