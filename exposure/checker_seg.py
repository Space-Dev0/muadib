"""Experimental foreground segmenter for renderer scenes; not a rule checker.

The model predicts only a foreground mask. It cannot yet recognize individual
objects, colors, shapes, spatial relationships, or issue a rule decision. A
segmentation score must not be reported as checker accuracy or attrition.
Torch is optional and loaded only for training/inference, not data generation.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from exposure.image_diagnostics import renderer_instance_map
from exposure.render import render_scene


def build_splits(seed, train_count, val_count, test_count):
    """Disjoint seed ranges; assert no duplicate pixels across any split."""
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if any(type(n) is not int or n < 1 for n in (train_count, val_count, test_count)):
        raise ValueError('split counts must be positive integers')
    result = {}
    hashes = set()
    next_seed = seed
    for name, count in [('train', train_count), ('validation', val_count), ('test', test_count)]:
        rows = []
        for scene_seed in range(next_seed, next_seed + count):
            scene = render_scene(scene_seed)
            pixels = np.asarray(scene.image.convert('RGB'), dtype=np.uint8).copy()
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError('duplicate scene pixels across data; use another seed range')
            hashes.add(digest)
            # Ground truth comes from object annotations, not a trained detector.
            mask = (renderer_instance_map(scene.objects) > 0).astype(np.uint8)
            rows.append({'seed': scene_seed, 'rgb': pixels, 'mask': mask, 'sha256': digest})
        result[name] = rows
        next_seed += count
    return result


def augment(rgb, seed, mode='train'):
    """Photometric perturbations; preserve geometric ground-truth masks."""
    if type(seed) is not int or seed < 0:
        raise ValueError('augmentation seed must be a nonnegative integer')
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), 'RGB')
    rng = random.Random(seed)
    if mode == 'train':
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 1.2)))
        sigma = rng.uniform(0.0, 12.0)
    elif mode == 'clean':
        sigma = 0
    elif mode == 'blur_1':
        image = image.filter(ImageFilter.GaussianBlur(1.0)); sigma = 0
    elif mode == 'brightness_1_1':
        image = ImageEnhance.Brightness(image).enhance(1.1); sigma = 0
    elif mode == 'noise_10':
        sigma = 10.0
    else:
        raise ValueError('unknown augmentation mode')
    arr = np.asarray(image, dtype=np.float32)
    if sigma:
        arr = arr + np.random.default_rng(seed).normal(0, sigma, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def binary_metrics(prediction, truth):
    """Aggregate pixel counts, preserving undefined denominator as None."""
    pred = np.asarray(prediction, dtype=bool)
    gold = np.asarray(truth, dtype=bool)
    if pred.shape != gold.shape or not pred.size:
        raise ValueError('prediction and truth must have same nonempty shape')
    tp = int(np.count_nonzero(pred & gold)); fp = int(np.count_nonzero(pred & ~gold))
    fn = int(np.count_nonzero(~pred & gold)); tn = int(np.count_nonzero(~pred & ~gold))
    ratio = lambda a, b: a / b if b else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'iou': ratio(tp, tp + fp + fn),
            'foreground_recall': ratio(tp, tp + fn),
            'background_false_positive_rate': ratio(fp, fp + tn)}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is required; install the optional train dependencies') from exc
    return torch


def build_model(width=16):
    """Three-level U-Net with two downsampling stages and a one-channel head."""
    if type(width) is not int or width < 2:
        raise ValueError('width must be an integer at least 2')
    torch = _torch()
    nn = torch.nn

    def block(inp, out):
        return nn.Sequential(nn.Conv2d(inp, out, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(out, out, 3, padding=1), nn.ReLU())

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = block(3, width); self.enc2 = block(width, width * 2)
            self.mid = block(width * 2, width * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, 2)
            self.dec2 = block(width * 4, width * 2)
            self.up1 = nn.ConvTranspose2d(width * 2, width, 2, 2)
            self.dec1 = block(width * 2, width)
            self.out = nn.Conv2d(width, 1, 1)

        def forward(self, x):
            a = self.enc1(x); b = self.enc2(self.pool(a)); c = self.mid(self.pool(b))
            y = self.dec2(torch.cat((self.up2(c), b), dim=1))
            return self.out(self.dec1(torch.cat((self.up1(y), a), dim=1)))
    return SmallUNet()


def _batches(rows, batch_size, *, mode, epoch, batch_seed):
    order = list(range(len(rows)))
    if mode == 'train':
        random.Random(batch_seed + epoch).shuffle(order)
    for first in range(0, len(order), batch_size):
        batch = [rows[i] for i in order[first:first + batch_size]]
        x = np.stack([augment(row['rgb'], batch_seed + epoch * 1_000_003 + row['seed'], mode)
                      for row in batch]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        y = np.stack([row['mask'] for row in batch])[:, None].astype(np.float32)
        yield x, y


def train(*, seed, counts, epochs, batch_size, width, device, output):
    """Train and evaluate segmentation only; use no held-out data for selection."""
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('epochs and batch size must be positive')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    output = Path(output)
    if output.exists():
        raise ValueError('output directory already exists; refusing overwrite')
    torch = _torch()
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    splits = build_splits(seed, *counts)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = build_model(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    def evaluate(rows, mode):
        model.eval()
        counts_out = {k: 0 for k in ('tp', 'fp', 'fn', 'tn')}
        with torch.no_grad():
            for x, y in _batches(rows, batch_size, mode=mode, epoch=0, batch_seed=seed + 2_000_000):
                logits = model(torch.from_numpy(x).to(device)).cpu().numpy()
                m = binary_metrics(logits >= 0, y >= .5)
                for key in counts_out:
                    counts_out[key] += m[key]
        tp, fp, fn, tn = (counts_out[k] for k in ('tp','fp','fn','tn'))
        counts_out.update({'iou': tp / (tp + fp + fn) if tp + fp + fn else None,
                           'foreground_recall': tp / (tp + fn) if tp + fn else None,
                           'background_false_positive_rate': fp / (fp + tn) if fp + tn else None})
        return counts_out

    best = -1.0; best_weights = None; history = []
    for epoch in range(epochs):
        model.train(); total_loss = 0.0; n = 0
        for x, y in _batches(splits['train'], batch_size, mode='train', epoch=epoch, batch_seed=seed + 1_000_000):
            x_tensor = torch.from_numpy(x).to(device); y_tensor = torch.from_numpy(y).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_tensor), y_tensor)
            loss.backward(); optimizer.step()
            total_loss += float(loss.item()) * len(x); n += len(x)
        validation = evaluate(splits['validation'], 'blur_1')
        history.append({'epoch': epoch + 1, 'mean_train_loss': total_loss / n, 'validation_blur_1': validation})
        if validation['iou'] is not None and validation['iou'] > best:
            best = validation['iou']
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_weights is None:
        raise RuntimeError('no valid validation score; refusing checkpoint')
    model.load_state_dict(best_weights)
    result = {'schema_version': 1, 'model': 'foreground_only_unet_3_level', 'width': width,
              'seed': seed, 'epochs': epochs, 'batch_size': batch_size, 'device': device,
              'pytorch_version': torch.__version__, 'learning_rate': 0.001,
              'split_seeds_and_pixel_sha256': {name: [{'seed': row['seed'], 'sha256': row['sha256']}
                                                     for row in rows] for name, rows in splits.items()},
              'selection': 'highest validation blur-1 pixel IoU; first epoch wins ties',
              'history': history,
              'test': {mode: evaluate(splits['test'], mode) for mode in ('clean', 'blur_1', 'brightness_1_1', 'noise_10')},
              'note': 'Experimental foreground segmentation only; no object/rule decisions, no checker attrition, no human-generated-image validation.'}
    output.mkdir(parents=True, exist_ok=False)
    torch.save({'state_dict': best_weights, 'width': width, 'kind': result['model']}, output / 'weights.pt')
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=98200)
    p.add_argument('--train-count', type=int, default=600)
    p.add_argument('--val-count', type=int, default=100)
    p.add_argument('--test-count', type=int, default=100)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args()
    try:
        result = train(seed=a.seed, counts=(a.train_count, a.val_count, a.test_count), epochs=a.epochs,
                       batch_size=a.batch_size, width=a.width, device=a.device, output=a.output)
    except (OSError, ValueError, RuntimeError) as exc:
        p.error(str(exc))
    print(json.dumps({'output': str(a.output), 'test': result['test'], 'note': result['note']}, indent=2))


if __name__ == '__main__':
    main()
