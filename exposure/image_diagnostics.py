"""CPU image-spectrum and explicitly paired, region-based error diagnostics.

Pixelwise errors are meaningful only when the supplied clean reference is the
actual aligned target for the prediction. Unconditional generation has no such
pixelwise target; do not compare it to an arbitrary clean render.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def _rgb(image):
    arr = np.asarray(image.convert('RGB'), dtype=np.float64) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError('image must have RGB channels')
    return arr


def radial_power_spectrum(image):
    """Mean-centered luminance power grouped by integer Fourier-radius bins.

    Luminance is Rec. 709 on gamma-encoded RGB (no inverse transfer), so this
    is a reproducible image diagnostic, not a physical light measurement.
    Spectrum power uses |FFT|^2/(H*W)^2; its sum equals pixel variance.
    """
    arr = _rgb(image)
    h, w = arr.shape[:2]
    if h < 2 or w < 2:
        raise ValueError('image must be at least 2 by 2')
    gray = arr @ np.array([0.2126, 0.7152, 0.0722])
    centered = gray - gray.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2 / (h * w) ** 2
    fy = np.fft.fftfreq(h) * h
    fx = np.fft.fftfreq(w) * w
    radius = np.floor(np.hypot(fy[:, None], fx[None, :])).astype(int)
    counts = np.bincount(radius.ravel())
    sums = np.bincount(radius.ravel(), weights=power.ravel())
    return {'luminance_mean': float(gray.mean()),
            'luminance_variance': float(np.mean(centered ** 2)),
            'total_power': float(power.sum()),
            'bins': [{'radius_bin': int(i), 'frequency_count': int(n),
                      'power_sum': float(s), 'mean_power': float(s / n)}
                     for i, (n, s) in enumerate(zip(counts, sums, strict=True)) if n]}


def masks_from_instances(labels, *, boundary_radius=1, gap_radius=8):
    """Disjoint interior, boundary ring and nearby inter-object gap masks.

    Labels must be a 2-D integer instance map: 0 background, distinct positive
    integers for objects. Boundary includes pixels within ``boundary_radius``
    on either side of an object's edge. Gap is background within ``gap_radius``
    of at least two different instances, excluding the boundary ring. Overlap
    in the labeled input cannot be represented and must be resolved upstream.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('instances must be a 2-D array of nonnegative integer labels')
    if type(boundary_radius) is not int or boundary_radius < 1:
        raise ValueError('boundary_radius must be a positive integer')
    if type(gap_radius) is not int or gap_radius < 1:
        raise ValueError('gap_radius must be a positive integer')
    ids = np.unique(labels[labels > 0])
    if not len(ids):
        raise ValueError('instance map contains no objects')
    structure = ndimage.generate_binary_structure(2, 1)
    foreground = labels > 0
    inner = np.zeros_like(foreground)
    boundary = np.zeros_like(foreground)
    nearby_count = np.zeros(labels.shape, dtype=np.int32)
    for obj in ids:
        mask = labels == obj
        eroded = ndimage.binary_erosion(mask, structure=structure, iterations=boundary_radius)
        dilated = ndimage.binary_dilation(mask, structure=structure, iterations=boundary_radius)
        inner |= eroded
        boundary |= dilated & ~eroded
        nearby_count += ndimage.distance_transform_edt(~mask) <= gap_radius
    # Narrow overlaps between dilated objects are boundary, even if another
    # instance's eroded interior reaches it; ensure strictly disjoint regions.
    interior = inner & ~boundary
    gap = (~foreground) & (nearby_count >= 2) & ~boundary
    return {'interior': interior, 'boundary': boundary, 'inter_object_gap': gap}


def region_mse(reference, prediction, labels, *, boundary_radius=1, gap_radius=8):
    """Per-pixel RGB MSE on aligned images; empty region returns None."""
    clean, predicted = _rgb(reference), _rgb(prediction)
    if clean.shape != predicted.shape or clean.shape[:2] != np.asarray(labels).shape:
        raise ValueError('reference, prediction, and instance map sizes must match')
    masks = masks_from_instances(labels, boundary_radius=boundary_radius, gap_radius=gap_radius)
    error = np.mean((predicted - clean) ** 2, axis=2)
    return {'overall_rgb_mse': float(error.mean()),
            'regions': {name: {'pixels': int(mask.sum()),
                               'rgb_mse': float(error[mask].mean()) if mask.any() else None}
                        for name, mask in masks.items()}}


def renderer_instance_map(objects):
    """Recover nonoverlapping prototype-renderer instances from annotations.

    Rendering each annotated object independently respects the renderer's actual
    shapes and antialiasing. Any overlapping derived masks are rejected instead
    of silently inventing ownership. This is not a general segmentation method.
    """
    from exposure.render import _draw, IMAGE_SIZE
    labels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
    for i, obj in enumerate(objects, 1):
        rgb = np.asarray(_draw((obj,)).image.convert('RGB'))
        mask = np.any(rgb < 250, axis=2)
        if np.any((labels > 0) & mask):
            raise ValueError('renderer-derived instance masks overlap; no unambiguous labels')
        labels[mask] = i
    return labels


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='image for spectral analysis')
    parser.add_argument('--reference', type=Path, help='pixel-aligned ground-truth reference')
    parser.add_argument('--instances', type=Path, help='integer PNG: 0 background, positive instance IDs')
    parser.add_argument('--assert-aligned', action='store_true', help='affirm image and reference share the same spatial target')
    parser.add_argument('--render-seed', type=int, help='make a controlled renderer/blur paired example instead of loading an image')
    parser.add_argument('--blur', type=float, default=1.0, help='Gaussian blur radius in controlled renderer example')
    parser.add_argument('--boundary-radius', type=int, default=1)
    parser.add_argument('--gap-radius', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.render_seed is not None:
            if args.image or args.reference or args.instances or args.assert_aligned:
                raise ValueError('--render-seed cannot be combined with image-pair arguments')
            if not math.isfinite(args.blur) or args.blur < 0:
                raise ValueError('blur must be finite and nonnegative')
            from exposure.render import render_scene
            scene = render_scene(args.render_seed)
            reference = scene.image
            labels = renderer_instance_map(scene.objects)
            prediction = reference.filter(ImageFilter.GaussianBlur(args.blur))
            source = {'type': 'controlled_renderer_blur', 'seed': args.render_seed,
                      'blur_radius': args.blur}
            paired = region_mse(reference, prediction, labels,
                                boundary_radius=args.boundary_radius, gap_radius=args.gap_radius)
        else:
            if not args.image:
                raise ValueError('--image or --render-seed is required')
            prediction = Image.open(args.image)
            source = {'type': 'image', 'image_sha256': _sha256(args.image)}
            paired = None
            if args.reference or args.instances or args.assert_aligned:
                if not (args.reference and args.instances and args.assert_aligned):
                    raise ValueError('paired MSE requires --reference, --instances, and --assert-aligned')
                reference = Image.open(args.reference)
                labels = np.asarray(Image.open(args.instances))
                paired = region_mse(reference, prediction, labels,
                                    boundary_radius=args.boundary_radius, gap_radius=args.gap_radius)
                source.update({'reference_sha256': _sha256(args.reference),
                               'instances_sha256': _sha256(args.instances),
                               'alignment': 'user asserted, not independently verified'})
        result = {'schema_version': 1, 'source': source,
                  'spectrum': radial_power_spectrum(prediction), 'paired_error': paired,
                  'parameters': {'boundary_radius_pixels': args.boundary_radius,
                                 'gap_radius_pixels': args.gap_radius},
                  'note': 'Descriptive diagnostics only. Region MSE needs a matched spatial target; spectral similarity does not establish semantic correctness.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
