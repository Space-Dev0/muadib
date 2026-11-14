"""Bounded one-cell multiscale diffusion learnability probe, not replication."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from exposure.mosaic_toy_train import load_cells
from exposure.mosaic_toy_sample import velocity_components


class MultiscaleVelocity(nn.Module):
    """Small two-level encoder-decoder with shared class/time conditioning."""
    def __init__(self, width=32):
        super().__init__()
        self.in_conv = nn.Conv2d(3, width, 3, padding=1)
        self.condition = nn.Embedding(2, width)
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.local = nn.Conv2d(width, width, 3, padding=1)
        self.down = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.middle = nn.Sequential(nn.SiLU(), nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU(), nn.Conv2d(width * 2, width * 2, 3, padding=1))
        self.up = nn.Conv2d(width * 2 + width, width, 3, padding=1)
        self.out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, image, label, t):
        cond = (self.condition(label) + self.time(t[:, None]))[:, :, None, None]
        local = F.silu(self.in_conv(image) + cond)
        local = local + F.silu(self.local(local))
        low = self.down(local)
        low = low + self.middle(low)
        up = F.interpolate(low, size=local.shape[-2:], mode='nearest')
        return self.out(F.silu(self.up(torch.cat((up, local), dim=1))))


@torch.no_grad()
def sample(model, label, seed, steps, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = torch.randn((1, 3, 64, 64), generator=generator, device=device)
    labels = torch.tensor([0 if label == 'red' else 1], device=device)
    times = torch.linspace(1, 0, steps, device=device)
    for i, t in enumerate(times):
        alpha = 0.99 - 0.98 * t
        clean, noise = velocity_components(image, model(image, labels, t.reshape(1)), alpha)
        if i + 1 == steps:
            image = clean
        else:
            next_alpha = 0.99 - 0.98 * times[i + 1]
            image = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
    if not torch.isfinite(image).all():
        raise ValueError('Non-finite generated image')
    pixels = ((image[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.asarray(pixels))


def run(dataset, output, *, seed=81322, width=32, batch_size=4, device='cuda',
        complexity='low', imbalance='balanced'):
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    if complexity not in ('low', 'high') or imbalance not in ('balanced', 'imbalanced'):
        raise ValueError('Invalid training cell')
    cells, manifest_hash = load_cells(dataset)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    model = MultiscaleVelocity(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    losses = []
    started = time.perf_counter()
    items = cells[(complexity, imbalance)]
    snapshots = []
    for step in range(1, 20001):
        examples = [items[rng.randrange(len(items))] for _ in range(batch_size)]
        clean = torch.stack([item[0] for item in examples]).to(device)
        labels = torch.tensor([item[1] for item in examples], device=device)
        t = torch.rand(batch_size, device=device)
        alpha = (0.99 - 0.98 * t)[:, None, None, None]
        noise = torch.randn_like(clean)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
        loss = F.mse_loss(model(noisy, labels, t), target)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step in (5000, 20000):
            checkpoint = output / f'multiscale-{step}.pt'
            torch.save({'kind': 'mosaic_multiscale_probe', 'schema_version': 1, 'width': width,
                        'step': step, 'seed': seed, 'cell': [complexity, imbalance], 'dataset_manifest_sha256': manifest_hash,
                        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, checkpoint)
            model.eval()
            for label in ('red', 'blue'):
                for sample_seed in (81330, 81331):
                    with (output / f'{step}-{label}-{sample_seed}.png').open('xb') as f:
                        sample(model, label, sample_seed, 50, device).save(f, format='PNG')
            model.train()
            snapshots.append({'step': step, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              'elapsed_seconds': time.perf_counter() - started,
                              'loss_mean_last_500': float(np.mean(losses[-500:]))})
    report = {'schema_version': 1, 'cell': [complexity, imbalance], 'training_seed': seed,
              'width': width, 'batch_size': batch_size, 'learning_rate': 0.0001, 'device': device,
              'torch_version': str(torch.__version__), 'python_version': sys.version.split()[0],
              'dataset_manifest_sha256': manifest_hash,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sample_seeds': [81330, 81331], 'sampler_steps': 50, 'snapshots': snapshots,
              'note': 'One-cell, one-seed feasibility probe; not architecture causality, MOSAIC replication or a gate.'}
    with (output / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--complexity', choices=('low', 'high'), default='low')
    parser.add_argument('--imbalance', choices=('balanced', 'imbalanced'), default='balanced')
    args = parser.parse_args()
    result = run(args.dataset, args.output, device=args.device, complexity=args.complexity, imbalance=args.imbalance)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
