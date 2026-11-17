"""Generate a fixed 64-image matched-cell evaluation from four 500k models.

One training-seed block, eight paired noise seeds and two target colors per
cell. This is an adapted exploratory contrast, not a Stage 0 gate decision.
"""
import argparse
import hashlib
import json
from pathlib import Path

from exposure.mosaic_distributed_supervisor import from_report
from exposure.mosaic_guidance_probe import run

CELLS = {
    'low-balanced': ('segment-500000-c7cb188b71', 'f702d0a2b8e292d92046e1b29520f03a3b6b8cd70dd2a26de9756b1466ef6ce5'),
    'low-imbalanced': ('segment-500000-779fd9aa13', '9d04966a3a5c2b4cbe86ee1b9c72ebcb7000db18a13f4793b8f4958384557957'),
    'high-balanced': ('segment-500000-6162927b4c', 'a4058a6464b3d35cc1c4f861c4c88841955fc67c40aa4d7976dd3b0a1b88e0a4'),
    'high-imbalanced': ('segment-500000-2f5efa0ec9', '85beb2facef0700af24b08e659db04a4b15b13060098f9386189d3994e48c27a'),
}
SEEDS = list(range(82000, 82008))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(workspace, output, *, device='cuda'):
    workspace = Path(workspace).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; preserve it and select a new path')
    inputs = {}
    for cell, (segment, expected) in CELLS.items():
        directory = workspace / 'outputs' / 'mosaic-distributed-81570' / cell / segment
        step, checkpoint, actual = from_report(directory, cell)
        if step != 500000 or actual != expected:
            raise ValueError(f'Unexpected checkpoint for {cell}')
        inputs[cell] = {'checkpoint': str(checkpoint), 'sha256': actual,
                        'training_report_sha256': sha(directory / 'report.json')}
    spec = {'schema_version': 1, 'cells': inputs, 'seeds': SEEDS,
            'target_colors': ['red', 'blue'], 'guidance': 1.0,
            'sampler': '1000-step DDPM, EMA, paired noise', 'training_seed': 81570,
            'metric': 'human-reviewed target-color circle presence, equal weighting by color',
            'secondary_metric': 'human-reviewed expected object count',
            'code_sha256': sha(__file__),
            'note': ('One matched training-seed block and 200 training images per cell. '
                     'Seeds are reused across conditions; no image is an independent training run. '
                     'Human labels and checker validation remain required.')}
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'protocol.json').open('x', encoding='utf-8') as f:
        json.dump(spec, f, indent=2, allow_nan=False); f.write('\n')
    for cell in CELLS:
        print(f'Sampling {cell}', flush=True)
        run(inputs[cell]['checkpoint'], output / cell, seeds=SEEDS, guidance=[1.0], device=device)
    with (output / 'COMPLETE').open('x', encoding='utf-8') as f:
        f.write('All four sampling reports completed.\n')
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = p.parse_args()
    try:
        generate(a.workspace, a.output, device=a.device)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
