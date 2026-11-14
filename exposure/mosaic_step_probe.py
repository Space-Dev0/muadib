"""Compare reverse-step counts from identical model weights and random seeds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from exposure.mosaic_multiscale_probe import MultiscaleVelocity, sample


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(checkpoint, output, *, device='cpu', seed=81330):
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    if device == 'cpu':
        torch.set_num_threads(1)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if cp.get('kind') != 'mosaic_multiscale_probe' or cp.get('step') != 20000:
        raise ValueError('Expected 20k multiscale probe checkpoint')
    model = MultiscaleVelocity(cp['width']).to(device).eval()
    model.load_state_dict(cp['state_dict'], strict=True)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for label in ('red', 'blue'):
        for steps in (50, 200, 500):
            image = sample(model, label, seed, steps, device)
            target = output / f'{label}-{seed}-{steps}.png'
            image.save(target)
            rows.append({'label': label, 'seed': seed, 'steps': steps, 'path': target.name,
                         'sha256': sha(target), 'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
    result = {'checkpoint_sha256': sha(checkpoint), 'checkpoint_step': cp['step'], 'cell': cp.get('cell'),
              'model_source_sha256': sha(Path(__import__('exposure.mosaic_multiscale_probe',fromlist=['x']).__file__)),
              'probe_source_sha256': sha(Path(__file__)), 'training_source_sha256': cp.get('source_sha256'),
              'device':device, 'torch_version':str(torch.__version__), 'results':rows,
              'note':'One checkpoint and seed per label; identical initial random seed and model. No training, no automatic quality metric or gate.'}
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--seed',type=int,default=81330)
    a=p.parse_args()
    result=run(a.checkpoint,a.output,device=a.device,seed=a.seed)
    print(json.dumps({'output':str(a.output),'images':len(result['results'])}))


if __name__ == '__main__':
    main()
