"""Bounded continuation of the multiscale feasibility probe; not a replication."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from exposure.mosaic_multiscale_probe import MultiscaleVelocity, sample
from exposure.mosaic_toy_train import load_cells


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(dataset, parent, output, *, steps, interval, seed, device, resume=None):
    if steps < 1 or interval < 1 or steps % interval or seed < 0:
        raise ValueError('Invalid steps, interval or seed')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Device unavailable')
    cells, manifest_hash = load_cells(dataset)
    parent_hash = digest(parent)
    parent_cp = torch.load(parent, map_location='cpu', weights_only=True)
    if (parent_cp.get('kind') != 'mosaic_multiscale_warmstart' or
        parent_cp.get('additional_updates') != 30000 or parent_cp.get('parent_step') != 20000 or
        parent_cp.get('dataset_manifest_sha256') != manifest_hash):
        raise ValueError('Expected the matching 20k plus 30k warm-start checkpoint')
    source_hash = digest(Path(__file__))
    model = MultiscaleVelocity(parent_cp['width']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    rng = random.Random(seed)
    start = 0
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        model.load_state_dict(parent_cp['state_dict'], strict=True)
        torch.manual_seed(seed)
        if device == 'cuda': torch.cuda.manual_seed_all(seed)
    else:
        output = output.resolve(strict=True)
        cp = torch.load(resume, map_location='cpu', weights_only=True)
        expected = (parent_hash, manifest_hash, source_hash, seed, device, str(torch.__version__))
        actual = tuple(cp.get(k) for k in ('parent_hash','manifest_hash','source_hash','seed','device','torch_version'))
        if cp.get('kind') != 'mosaic_resumable_probe' or actual != expected:
            raise ValueError('Resume provenance or runtime mismatch')
        start = cp['step']
        if start >= steps or start % interval:
            raise ValueError('Resume checkpoint is outside the requested interval')
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        torch.set_rng_state(cp['torch_rng'])
        if device == 'cuda': torch.cuda.set_rng_state_all(cp['cuda_rng'])
        rng.setstate(cp['python_rng'])
    items = cells[('low', 'balanced')]
    loss_window = []
    for step in range(start + 1, steps + 1):
        batch = [items[rng.randrange(len(items))] for _ in range(4)]
        clean = torch.stack([x[0] for x in batch]).to(device)
        labels = torch.tensor([x[1] for x in batch], device=device)
        t = torch.rand(4, device=device)
        alpha = (0.99 - 0.98*t)[:, None, None, None]
        noise = torch.randn_like(clean)
        noisy = alpha.sqrt()*clean + (1-alpha).sqrt()*noise
        target = alpha.sqrt()*noise - (1-alpha).sqrt()*clean
        loss = F.mse_loss(model(noisy, labels, t), target)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_window.append(float(loss.detach().cpu()))
        if step % interval == 0:
            path = output / f'resumable-plus-{step}.pt'
            if path.exists(): raise FileExistsError(path)
            state = {'kind':'mosaic_resumable_probe', 'step':step, 'parent_hash':parent_hash,
                     'manifest_hash':manifest_hash, 'source_hash':source_hash, 'seed':seed,
                     'device':device, 'torch_version':str(torch.__version__),
                     'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                     'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                     'python_rng':rng.getstate()}
            temp = output / f'.resumable-plus-{step}.tmp'
            if temp.exists(): raise FileExistsError(temp)
            torch.save(state, temp)
            os.replace(temp, path)
            model.eval()
            for label in ('red','blue'):
                for sample_seed in (81330,81331):
                    target = output / f'plus-{step}-{label}-{sample_seed}.png'
                    with target.open('xb') as f: sample(model,label,sample_seed,50,device).save(f,format='PNG')
            model.train()
            record = {'step':step,'parent_sha256':parent_hash,'checkpoint_sha256':digest(path),
                      'manifest_sha256':manifest_hash,'source_sha256':source_hash,'seed':seed,'device':device,
                      'torch_version':str(torch.__version__),'loss_last_500':sum(loss_window[-500:])/len(loss_window[-500:]),
                      'note':'Fresh AdamW/RNG at run start; one cell, no gate. Exact resume requires same runtime and compatible deterministic backend.'}
            with (output / f'plus-{step}.json').open('x') as f: json.dump(record,f,indent=2,allow_nan=False)
            print(json.dumps(record), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--interval',type=int,default=10000)
    p.add_argument('--seed',type=int,default=81360);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args()
    train(a.dataset,a.parent,a.output,steps=a.steps,interval=a.interval,seed=a.seed,device=a.device,resume=a.resume)

if __name__ == '__main__': main()
