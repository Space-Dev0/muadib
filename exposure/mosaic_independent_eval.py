"""Resume fixed inference and freeze review packs for model-seed block 82600."""
import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from exposure import mosaic_distributed_supervisor as supervisor
from exposure.mosaic_guidance_probe import run
from exposure.mosaic_matched_eval_pack import build

CELLS = ('low-balanced', 'low-imbalanced', 'high-balanced', 'high-imbalanced')
SEEDS = list(range(83000, 83008))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_report(directory, expected_checkpoint):
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    rows = report['images']
    expected = {(color, seed, 1.0) for color in ('red', 'blue') for seed in SEEDS}
    keys = [(row['color'], row['seed'], row['guidance']) for row in rows]
    if (report['checkpoint_sha256'] != expected_checkpoint or report['seeds'] != SEEDS or
            report['guidance'] != [1.0] or len(keys) != 16 or set(keys) != expected):
        raise ValueError(f'Unexpected sample report: {directory}')
    for row in rows:
        if Path(row['file']).name != row['file'] or sha(directory / row['file']) != row['sha256']:
            raise ValueError(f'Sample integrity failure: {directory}')
    return report


def checkpoints(workspace):
    supervisor.RUN_ID = 82600
    root = workspace / 'outputs' / 'mosaic-distributed-82600'
    found = {}
    for cell in CELLS:
        candidates = []
        for directory in (root / cell).glob('segment-500000-*'):
            if (directory / 'report.json').is_file():
                candidates.append(supervisor.from_report(directory, cell))
        if len(candidates) != 1 or candidates[0][0] != 500000:
            raise ValueError(f'Expected one verified 500k report for {cell}')
        _, checkpoint, digest = candidates[0]
        found[cell] = {'checkpoint': str(checkpoint), 'sha256': digest,
                       'training_report_sha256': sha(checkpoint.parent / 'report.json')}
    return found


def generate(workspace, output, review, *, device='cuda'):
    workspace = Path(workspace).resolve(strict=True)
    output = Path(output).resolve(); review = Path(review).resolve()
    if output == review or output.is_relative_to(review) or review.is_relative_to(output):
        raise ValueError('Output and review directories must be separate')
    inputs = checkpoints(workspace)
    spec = {'schema_version': 1, 'cells': inputs, 'seeds': SEEDS,
            'target_colors': ['red', 'blue'], 'guidance': 1.0,
            'sampler': '1000-step DDPM, EMA, paired noise', 'training_seed': 82600,
            'metric': 'human-reviewed target-color circle presence, equal weighting by color',
            'secondary_metric': 'human-reviewed expected object count', 'code_sha256': sha(__file__),
            'note': 'One model-seed block on shared fixed data; human review and checker validation remain required.'}
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        (output / '.owned').write_text('mosaic independent evaluation 82600\n', encoding='utf-8')
    elif not (output / '.owned').is_file():
        raise ValueError('Existing output is not owned by this evaluation')
    protocol = output / 'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text(encoding='utf-8')) != spec:
            raise ValueError('Frozen protocol or checkpoint changed')
    else:
        temporary = output / ('protocol-' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(spec, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, protocol)
    for cell in CELLS:
        final = output / cell
        if final.exists():
            valid_report(final, inputs[cell]['sha256']); continue
        parent = output / 'attempts' / cell; parent.mkdir(parents=True, exist_ok=True)
        completed = None
        for attempt in sorted(parent.iterdir()):
            if (attempt / 'report.json').is_file():
                try:
                    valid_report(attempt, inputs[cell]['sha256']); completed = attempt; break
                except (OSError, ValueError, KeyError):
                    pass
        if completed is None:
            completed = parent / ('attempt-' + uuid.uuid4().hex[:10])
            print(f'Sampling {cell}; preserving partial attempts', flush=True)
            run(inputs[cell]['checkpoint'], completed, seeds=SEEDS, guidance=[1.0], device=device)
            valid_report(completed, inputs[cell]['sha256'])
        staged = output / (cell + '-stage-' + uuid.uuid4().hex[:10])
        shutil.copytree(completed, staged)
        os.replace(staged, final)
    complete = output / 'COMPLETE'
    if not complete.exists():
        complete.write_text('All four sampling reports completed.\n', encoding='utf-8')
    if review.exists():
        # Existing packs are never overwritten; a partial pack must be inspected.
        if not (review / 'private' / 'archive-hashes.json').is_file():
            raise ValueError('Existing review pack is incomplete; preserve and inspect it')
    else:
        staging = review.parent / (review.name + '-stage-' + uuid.uuid4().hex[:10])
        build(output, staging, seeds=SEEDS)
        os.replace(staging, review)
    print(f'Evaluation and blinded review packs ready: {review}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    try:
        generate(args.workspace, args.output, args.review, device=args.device)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
