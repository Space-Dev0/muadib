"""Paired classifier-free-guidance sampling from a frozen PilotUNet checkpoint.

Exploratory diagnostic: not a calibrated evaluator, MOSAIC replication or gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from exposure import pilot_train
from exposure.identity_eps_finetune import _sample
from exposure.mosaic_pilotunet_warmstart import load_warmstart_checkpoint
from exposure.velocity_probe import velocity_to_noise


class GuidedVelocityAsNoise(torch.nn.Module):
    """Combine conditional/unconditional velocity then convert it to epsilon."""
    def __init__(self, model, alpha_bar, guidance):
        super().__init__()
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError('guidance must be finite and nonnegative')
        self.model = model
        self.register_buffer('alpha_bar', alpha_bar)
        self.guidance = guidance

    def forward(self, noisy, timestep, tokens, drop=None):
        conditional = self.model(noisy, timestep, tokens, None)
        if self.guidance == 1:
            velocity = conditional
        else:
            unconditional = self.model(noisy, timestep, tokens,
                                       torch.ones(noisy.shape[0], dtype=torch.bool, device=noisy.device))
            velocity = unconditional + self.guidance * (conditional - unconditional)
        alpha = self.alpha_bar[timestep].view(-1, 1, 1, 1)
        return velocity_to_noise(noisy, velocity, alpha)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(checkpoint, output, *, seeds, guidance, device):
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct nonnegative integers')
    if not guidance or any(not math.isfinite(g) or g < 0 for g in guidance) or len(set(guidance)) != len(guidance):
        raise ValueError('guidance values must be distinct finite nonnegative numbers')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Device unavailable')
    if device == 'cpu':
        torch.set_num_threads(1)
    checkpoint = Path(checkpoint)
    cp = load_warmstart_checkpoint(checkpoint)
    target = torch.device(device)
    config = cp['config']
    pilot_train._configure_determinism(bool(config.get('deterministic', False)), device)
    rules = pilot_train.load_rules(cp['rules'])
    names = list(rules)
    if set(names) != {'red', 'blue'}:
        raise ValueError('Expected exactly red and blue rules')
    tokens = pilot_train.encode_rules(rules).to(target)
    model = pilot_train.PilotUNet(config['width'], config['timesteps']).to(target).eval()
    model.load_state_dict(cp['ema'], strict=True)
    beta, alpha_bar = pilot_train.schedule(config['timesteps'], target)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for scale in guidance:
        predictor = GuidedVelocityAsNoise(model, alpha_bar, scale).to(target).eval()
        for seed in seeds:
            for color in ('red', 'blue'):
                image = _sample(predictor, 'standard', tokens=tokens[names.index(color):names.index(color)+1],
                                beta=beta, alpha_bar=alpha_bar, seed=seed, device=target)
                filename = f'{color}-{seed}-guidance-{scale:g}.png'
                path = output / filename
                with path.open('xb') as file:
                    image.save(file, format='PNG')
                rows.append({'color': color, 'seed': seed, 'guidance': scale, 'file': filename,
                             'sha256': digest(path), 'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
    report = {'schema_version': 1, 'checkpoint_sha256': digest(checkpoint),
              'probe_sha256': digest(__file__), 'model_source_sha256': digest(pilot_train.__file__),
              'device': device, 'torch_version': str(torch.__version__), 'seeds': seeds,
              'guidance': guidance, 'sampler': '1000-step DDPM, EMA, paired noise', 'images': rows,
              'note': ('Exploratory fixed-checkpoint guidance comparison; images and seeds are dependent, '
                       'not an accuracy estimate, independent validation or Stage 0 gate. Higher guidance '
                       'can amplify artifacts and changes the sampling distribution.')}
    with (output / 'report.json').open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[81380, 81381])
    parser.add_argument('--guidance', type=float, nargs='+', default=[1., 2., 4.])
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.checkpoint, args.output, seeds=args.seeds,
                             guidance=args.guidance, device=args.device), indent=2))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
