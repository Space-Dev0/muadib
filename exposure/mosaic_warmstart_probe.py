"""Bounded weight-only warm start for one multiscale feasibility model.

The parent checkpoint does not contain optimizer or RNG state: this is a new
optimizer run, not exact resume, and is not a MOSAIC replication.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from exposure.mosaic_multiscale_probe import MultiscaleVelocity, sample
from exposure.mosaic_toy_train import load_cells


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(dataset, checkpoint_path, output, *, additional_updates, save_every, seed,
        device, sampler_steps=50):
    if any(type(v) is not int for v in (additional_updates, save_every, seed, sampler_steps)):
        raise ValueError('Update counts, sample steps and seed must be integers')
    if additional_updates <= 0 or save_every <= 0 or additional_updates % save_every or seed < 0 or sampler_steps < 2:
        raise ValueError('Updates must be positive and divisible by save_every; seed must be nonnegative and sampler steps >=2')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    cells, manifest_hash = load_cells(dataset)
    parent_hash = digest(checkpoint_path)
    cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (not isinstance(cp, dict) or cp.get('kind') != 'mosaic_multiscale_probe'
            or cp.get('schema_version') != 1 or cp.get('step') != 20000
            or cp.get('dataset_manifest_sha256') != manifest_hash):
        raise ValueError('Expected a 20k multiscale checkpoint matching the dataset')
    # The original low-balanced checkpoint predates the explicit cell field.
    if cp.get('cell', ['low', 'balanced']) != ['low', 'balanced']:
        raise ValueError('Expected the low-balanced checkpoint')
    width = cp.get('width')
    if type(width) is not int or width <= 0:
        raise ValueError('Invalid checkpoint width')
    model = MultiscaleVelocity(width).to(device)
    model.load_state_dict(cp['state_dict'], strict=True)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    items = cells[('low', 'balanced')]
    batch_size = 4
    losses, snapshots = [], []
    started = time.perf_counter()
    for step in range(1, additional_updates + 1):
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
        if step % save_every == 0:
            path = output / f'warmstart-plus-{step}.pt'
            torch.save({'schema_version': 1, 'kind': 'mosaic_multiscale_warmstart',
                        'width': width, 'additional_updates': step, 'parent_step': 20000,
                        'parent_checkpoint_sha256': parent_hash, 'dataset_manifest_sha256': manifest_hash,
                        'source_sha256': digest(Path(__file__)), 'seed': seed,
                        'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, path)
            model.eval()
            for label in ('red', 'blue'):
                for sample_seed in (81330, 81331):
                    with (output / f'plus-{step}-{label}-{sample_seed}.png').open('xb') as f:
                        sample(model, label, sample_seed, sampler_steps, device).save(f, format='PNG')
            model.train()
            snapshots.append({'additional_updates': step, 'checkpoint_sha256': digest(path),
                              'elapsed_seconds': time.perf_counter() - started,
                              'loss_mean_last_500': float(np.mean(losses[-500:]))})
    report = {'schema_version': 1, 'kind': 'weight-only-warm-start', 'parent_step': 20000,
              'parent_checkpoint_sha256': parent_hash, 'parent_checkpoint_source_sha256': cp.get('source_sha256'),
              'dataset_manifest_sha256': manifest_hash, 'training_seed': seed,
              'source_sha256': digest(Path(__file__)), 'width': width, 'batch_size': batch_size,
              'learning_rate': 0.0001, 'device': device, 'torch_version': str(torch.__version__),
              'python_version': sys.version.split()[0], 'sample_seeds': [81330, 81331],
              'sampler_steps': sampler_steps, 'snapshots': snapshots,
              'note': 'New optimizer and RNG; not exact resume. Single-cell one-seed feasibility only; not a MOSAIC replication or Stage 0 gate.'}
    with (output / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--additional-updates', type=int, default=20000)
    p.add_argument('--save-every', type=int, default=10000)
    p.add_argument('--seed', type=int, default=81350)
    p.add_argument('--sampler-steps', type=int, default=50)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = p.parse_args()
    try:
        report = run(a.dataset, a.checkpoint, a.output, additional_updates=a.additional_updates,
                     save_every=a.save_every, seed=a.seed, device=a.device, sampler_steps=a.sampler_steps)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        p.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
