"""Diagnose palette-versus-geometry abstention; never use as a research checker.

Loose tolerances are deliberately unsafe interventions for development only.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image
from exposure import checker
from exposure.audit_pilot_score import evaluate as verify_pilot
from exposure.human_audit import read_reviews
from exposure.render import parse_rule
from exposure.rules import Attribute

TOLERANCES = (40, 45, 60, 100, 1000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_image(image, predicates, tolerance):
    """Apply the existing detector at an unsafe diagnostic tolerance."""
    try:
        flags = []
        objects = checker._detect_palette(image, palette_tolerance=tolerance,
                    narrow_triangles=True, guard_spatial=False, uncertain_out=flags,
                    component_counts=(2, 3, 4))
        result = checker._evaluate_uncertain(objects, flags, predicates)
        if len(objects) != 3 and result is False:
            result = None
        return result, 'answered' if result is not None else 'uncertain'
    except checker.DetectionError as exc:
        message = str(exc)
        if message == 'Unrecognized foreground colors':
            category = 'palette'
        elif message.startswith('Expected two, three or four separated objects; found '):
            category = 'component_count'
        elif message == 'Unsupported object size or occlusion':
            category = 'size_or_occlusion'
        elif 'shape' in message:
            category = 'shape'
        elif 'color' in message:
            category = 'color'
        else:
            category = 'other_detection_failure'
        return None, category


def evaluate(pack, kind):
    """Summarize intervention outcomes, optionally using existing human labels."""
    private = pack / 'private'
    labels = None
    if kind == 'reviewed':
        reviews = {i: (pack / f'reviewer-{i}' / 'rule-reviews.csv',
                       pack / f'reviewer-{i}' / 'predicate-reviews.csv') for i in range(1, 5)}
        verified = verify_pilot(pack, reviews)
        if verified['shared_rule_agreement']['disagreements']:
            raise ValueError('Shared human labels are disputed')
        labels = {}
        for rule_path, _ in reviews.values():
            for identifier, value in read_reviews(rule_path).items():
                if identifier in labels and labels[identifier] != value:
                    raise ValueError('Conflicting human labels')
                labels[identifier] = value
        manifest = json.loads((private / 'all-rules.json').read_text())['samples']
        rows = [(r['sample_id'], private / r['image_path'], parse_rule(r['predicates'])) for r in manifest]
        if set(labels) != {r[0] for r in rows}:
            raise ValueError('Human labels do not match manifest')
    elif kind == 'mosaic':
        manifest = json.loads((private / 'provenance.json').read_text())['items']
        rows = [(identifier, private / 'images' / f'{identifier}.png',
                (Attribute('a', 'color', row['label']), Attribute('a', 'shape', 'circle')))
                for identifier, row in manifest.items()]
        for identifier, path, _ in rows:
            if digest(path) != manifest[identifier]['image_sha256']:
                raise ValueError(f'Image hash mismatch: {identifier}')
    else:
        raise ValueError('Unknown pack kind')
    summary = defaultdict(Counter)
    image_hashes = {}
    for identifier, path, predicates in rows:
        image_hashes[identifier] = digest(path)
        with Image.open(path) as image:
            for tolerance in TOLERANCES:
                pred, status = probe_image(image, predicates, tolerance)
                summary[tolerance][status] += 1
                if pred is not None and labels is not None:
                    summary[tolerance]['accepted_disagreements'] += pred != labels[identifier]
    return {'schema_version': 1, 'kind': kind, 'images': len(rows),
            'checker_source_sha256': digest(Path(checker.__file__)),
            'probe_source_sha256': digest(Path(__file__)), 'image_sha256': image_hashes,
            'summary': {str(k): dict(v) for k, v in sorted(summary.items())},
            'note': ('Diagnostic intervention only. 1000 effectively removes the palette guard; '
                     'remaining geometry errors are the first encountered failures, not exhaustive. '
                     'Previously reviewed labels now inform development, not independent validation. '
                     'No model performance, checker accuracy guarantee or Stage 0 gate is inferred.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--kind', choices=('reviewed', 'mosaic'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.pack, args.kind)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write('\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
