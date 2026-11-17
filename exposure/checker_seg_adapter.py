"""Frozen experimental rule adapter for a foreground-only segmenter.

Component geometry, colors and relation centers are heuristic; a binary mask
cannot distinguish merged/overlapping objects in general. Results require an
independent human audit. This is not a replacement for the standard checker.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from exposure.checker import DetectionError, audit_rules
from exposure.checker_seg import _torch, build_model
from exposure.render import COLORS, IMAGE_SIZE, parse_rule
from exposure.rules import SceneObject, satisfies


# Fixed before applying this adapter to human-reviewed generated images.
CONFIG = {'logit_threshold': 0.0, 'accepted_component_counts': [3],
          'ignore_component_below_pixels': 5, 'min_component_pixels': 30,
          'width_range': [8, 16], 'height_range': [8, 16],
          'shape_fill_intervals': {'triangle': [0.43, 0.62], 'circle': [0.68, 0.90], 'square': [0.94, 1.0]},
          'palette_max_distance': 100.0, 'palette_min_margin': 20.0,
          'spatial_min_center_gap': 2.0}


def objects_from_mask(image, mask):
    """Conservatively parse three separate, renderer-sized components."""
    if image.mode != 'RGB' or image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise DetectionError('Expected 64x64 RGB image')
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise DetectionError('Invalid mask dimensions')
    labels, n = ndimage.label(mask)
    ids = []
    for i in range(1, n + 1):
        count = int((labels == i).sum())
        if count < CONFIG['ignore_component_below_pixels']:
            continue
        if count < CONFIG['min_component_pixels']:
            raise DetectionError('Ambiguous foreground fragment')
        ids.append(i)
    if len(ids) not in CONFIG['accepted_component_counts']:
        raise DetectionError('Ambiguous number of separated objects')
    rgb = np.asarray(image, dtype=np.float64)
    palette = np.asarray(list(COLORS.values()), dtype=np.float64)
    found = []
    for i in ids:
        ys, xs = np.nonzero(labels == i)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        width, height = x1-x0+1, y1-y0+1
        if not (CONFIG['width_range'][0] <= width <= CONFIG['width_range'][1] and
                CONFIG['height_range'][0] <= height <= CONFIG['height_range'][1]):
            raise DetectionError('Unsupported component size or overlap')
        fill = len(xs) / (width * height)
        shapes = [name for name, (lo, hi) in CONFIG['shape_fill_intervals'].items() if lo <= fill <= hi]
        if len(shapes) != 1:
            raise DetectionError('Ambiguous component shape')
        # Median is robust to moderate edges but has no calibrated generated-image guarantee.
        color = np.median(rgb[ys, xs], axis=0)
        dist = np.linalg.norm(palette - color, axis=1)
        ordered = np.argsort(dist)
        if dist[ordered[0]] > CONFIG['palette_max_distance'] or dist[ordered[1]] - dist[ordered[0]] < CONFIG['palette_min_margin']:
            raise DetectionError('Ambiguous component color')
        found.append(SceneObject(tuple(COLORS)[ordered[0]], shapes[0], (x0+x1)/2, (y0+y1)/2))
    return tuple(found)


def classify(image, mask, predicates):
    objects = objects_from_mask(image, mask)
    # Without calibrated center error, near ties in either coordinate abstain.
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            if abs(a.x-b.x) <= CONFIG['spatial_min_center_gap'] or abs(a.y-b.y) <= CONFIG['spatial_min_center_gap']:
                raise DetectionError('Ambiguous spatial ordering')
    return bool(satisfies(objects, predicates))


def load_model(checkpoint, expected_sha256, device):
    torch = _torch()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Checkpoint hash does not match frozen expected SHA-256')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get('kind') != 'foreground_only_unet_3_level':
        raise ValueError('Unexpected segmentation model kind')
    model = build_model(payload['width']).to(device)
    model.load_state_dict(payload['state_dict']); model.eval()
    return model


def infer_mask(image, model, device):
    torch = _torch()
    data = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2,0,1)[None] / 255
    with torch.no_grad():
        logits = model(torch.from_numpy(data).to(device)).cpu().numpy()[0,0]
    return logits >= CONFIG['logit_threshold']


def predict_synthetic(model, device, start, count, case):
    from exposure.render import render_scene
    rules = audit_rules()
    stats = {'total': count, 'accepted': 0, 'abstained': 0, 'false_positive': 0,
             'false_negative': 0, 'rule_decisions': 0, 'reasons': {}, 'case': case}
    for seed in range(start, start+count):
        scene = render_scene(seed)
        image = scene.image if case == 'clean' else scene.image.filter(ImageFilter.GaussianBlur(1.0))
        try:
            predicted = objects_from_mask(image, infer_mask(image, model, device))
            for i,a in enumerate(predicted):
                for b in predicted[i+1:]:
                    if abs(a.x-b.x) <= 2 or abs(a.y-b.y) <= 2:
                        raise DetectionError('Ambiguous spatial ordering')
        except DetectionError as exc:
            stats['abstained'] += 1
            reason = str(exc); stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
            continue
        stats['accepted'] += 1
        for rule in rules.values():
            truth = satisfies(scene.objects, rule); value = satisfies(predicted, rule)
            stats['false_positive'] += int(value and not truth)
            stats['false_negative'] += int(truth and not value)
            stats['rule_decisions'] += 1
    return stats


def predict_manifest(model, device, pack, manifest):
    data = json.loads(manifest.read_text(encoding='utf-8'))
    predictions = []
    for sample in data['samples']:
        identifier = sample['sample_id']
        path = pack / f'{identifier}.png'
        image = Image.open(path).convert('RGB')
        try:
            decision = classify(image, infer_mask(image, model, device), parse_rule(sample['predicates']))
            reason = None
        except DetectionError as exc:
            decision, reason = None, str(exc)
        predictions.append({'sample_id': identifier, 'decision': decision, 'reason': reason,
                            'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True); p.add_argument('--expected-weights-sha256', required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--synthetic-start', type=int); p.add_argument('--synthetic-count', type=int, default=100)
    p.add_argument('--synthetic-case', choices=('clean', 'blur_1'), default='clean')
    p.add_argument('--images', type=Path); p.add_argument('--manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        if (a.synthetic_start is None) == (a.images is None or a.manifest is None):
            raise ValueError('Choose synthetic start OR both images and manifest')
        if a.synthetic_start is not None and (a.synthetic_start < 0 or a.synthetic_count < 1):
            raise ValueError('Invalid synthetic range')
        model = load_model(a.weights, a.expected_weights_sha256, a.device)
        result = {'schema_version': 1, 'checkpoint_sha256': a.expected_weights_sha256,
                  'adapter_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'config': CONFIG, 'note': 'Exploratory frozen binary-foreground adapter; accepted predictions may still be wrong.'}
        if a.synthetic_start is not None:
            result.update({'synthetic_start': a.synthetic_start,
                           'synthetic': predict_synthetic(model, a.device, a.synthetic_start, a.synthetic_count, a.synthetic_case)})
        else:
            result['manifest_sha256'] = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
            result['predictions'] = predict_manifest(model, a.device, a.images, a.manifest)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
        print(f'Wrote {a.output}')
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
