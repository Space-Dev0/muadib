"""Resume matched-cell GPU training in verified bounded 10k segments on Windows."""
import argparse
import hashlib
import importlib
import json
import os
import pickle
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
from exposure.mosaic_pilotunet_probe import load_checkpoint as load_initial_checkpoint
from exposure.mosaic_pilotunet_warmstart import load_warmstart_checkpoint

CELLS = {'low-balanced': ('low', 'balanced'), 'low-imbalanced': ('low', 'imbalanced'),
         'high-balanced': ('high', 'balanced'), 'high-imbalanced': ('high', 'imbalanced')}
MODULES = ('pilot_train', 'mosaic_pilotunet_probe', 'mosaic_pilotunet_warmstart',
           'mosaic_toy_train', 'velocity_probe', 'identity_eps_finetune', 'measure',
           'render', 'rules', 'mosaic_matched_cell')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(path, cell, step, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    cp = load_warmstart_checkpoint(path)
    config = cp['config']
    expected = {'width': 16, 'timesteps': 1000, 'batch_size': 4, 'seed': 81571,
                'lr': 0.0001, 'dropout': 0.1, 'ema_decay': 0.999, 'deterministic': False}
    if (cp['cell'] != list(CELLS[cell]) or cp['report']['cumulative_step'] != step or
            any(config.get(k) != v for k, v in expected.items())):
        raise ValueError(f'Unexpected checkpoint configuration: {path}')
    if (not torch.cuda.is_available() or cp['rng']['device'] != 'cuda' or
            len(cp['rng']['torch_cuda']) != torch.cuda.device_count() or
            cp['provenance']['torch'] != str(torch.__version__) or
            tuple(cp['provenance']['runtime']['python']) != sys.version_info[:3]):
        raise ValueError('Incompatible CUDA, PyTorch or Python major/minor runtime')
    recorded = cp['provenance']['runtime']['code_sha256']
    for name in MODULES:
        module = importlib.import_module('exposure.' + name)
        if recorded.get(name) != digest(module.__file__):
            raise ValueError(f'Checkpoint source mismatch: {name}')
    return cp


def from_report(directory, cell):
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    snapshot = max(report['snapshots'], key=lambda row: row['step'])
    step = snapshot['step']
    path = directory / f'warmstart-{step:06d}.pt'
    checkpoint(path, cell, step, snapshot['checkpoint_sha256'])
    return step, path.resolve(), snapshot['checkpoint_sha256']


def initial_from_report(directory, cell):
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    snapshot = max(report['snapshots'], key=lambda row: row['step'])
    step = snapshot['step']
    path = directory / f'pilotunet-{step:06d}.pt'
    if step != 1000 or digest(path) != snapshot['checkpoint_sha256']:
        raise ValueError(f'Invalid initial checkpoint: {path}')
    cp = load_initial_checkpoint(path)
    expected = {'seed': 81570, 'width': 16, 'timesteps': 1000, 'batch_size': 4,
                'lr': 0.0001, 'dropout': 0.1, 'ema_decay': 0.999, 'deterministic': False}
    if (cp['cell'] != list(CELLS[cell]) or cp['report']['step'] != 1000 or
            any(cp['config'].get(k) != v for k, v in expected.items())):
        raise ValueError('Initial checkpoint configuration mismatch')
    recorded = cp['provenance']['runtime']['code_sha256']
    for name in MODULES:
        # The initial probe does not import the warm-start trainer.
        if name == 'mosaic_pilotunet_warmstart':
            continue
        if recorded.get(name) != digest(importlib.import_module('exposure.' + name).__file__):
            raise ValueError(f'Initial checkpoint source mismatch: {name}')
    return step, path.resolve(), snapshot['checkpoint_sha256']


def latest(workspace, root, cell):
    source = workspace / 'outputs' / f'mosaic-matched-81570-{cell}-continued'
    candidates = []
    initial = workspace / 'outputs' / f'mosaic-matched-81570-{cell}-initial'
    for directory in [initial, *(root / cell).glob('initial-*')]:
        if (directory / 'report.json').is_file():
            candidates.append((*initial_from_report(directory, cell), 'warmstart'))
    for directory in [source, *(root / cell).glob('segment-*')]:
        if (directory / 'report.json').is_file():
            candidates.append((*from_report(directory, cell), 'resume'))
        elif directory.is_dir():
            # A power loss may leave a complete checkpoint but no final report.
            # There is no independent recorded hash; validate content and record
            # its observed hash in the log before resuming. Partial files fail load.
            for path in directory.glob('warmstart-*.pt'):
                try:
                    step = int(path.stem.removeprefix('warmstart-'))
                    sha = digest(path)
                    checkpoint(path, cell, step, sha)
                    print(f'Recovered loadable checkpoint without report hash: {path}, sha256={sha}', flush=True)
                    candidates.append((step, path.resolve(), sha, 'resume'))
                except (ValueError, OSError, EOFError, RuntimeError, KeyError, pickle.UnpicklingError) as exc:
                    print(f'Skipping invalid checkpoint {path}: {exc}', flush=True)
    return max(candidates, key=lambda row: row[0]) if candidates else None


def foreign_training_running():
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' "
               "-and $_.CommandLine -match 'exposure\\.mosaic_matched_cell' } | Select-Object -ExpandProperty ProcessId")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True,
                            text=True, check=True, timeout=45)
    return bool(result.stdout.strip())


def predecessor_running(name):
    if not name:
        return False
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                             f"(Get-ScheduledTask -TaskName '{name}' -ErrorAction Stop).State"],
                            capture_output=True, text=True, check=True, timeout=45)
    return result.stdout.strip() in ('Running', 'Queued')


def supervise(workspace, cells, target, predecessor):
    workspace = Path(workspace).resolve(strict=True)
    root = workspace / 'outputs' / 'mosaic-distributed-81570'
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'STOP'
    while predecessor_running(predecessor):
        if marker.exists():
            return
        time.sleep(30)
    if foreign_training_running():
        raise RuntimeError('A matched-cell training process is already running; refusing overlap')
    for cell in cells:
        while not marker.exists():
            found = latest(workspace, root, cell)
            if found is None:
                out = root / cell / f'initial-{uuid.uuid4().hex[:10]}'
                args = [sys.executable, '-u', '-m', 'exposure.mosaic_matched_cell',
                        '--complexity', CELLS[cell][0], '--imbalance', CELLS[cell][1],
                        '--trainer', 'probe', 'bounded', '--dataset',
                        str(workspace / 'outputs' / 'mosaic-data-gpu-feasibility-81300'),
                        '--output', str(out), '--seed', '81570', '--steps', '1000',
                        '--batch-size', '4', '--width', '16', '--timesteps', '1000',
                        '--lr', '0.0001', '--dropout', '0.1', '--ema-decay', '0.999',
                        '--preview-seeds', '81580', '--device', 'cuda']
                print(f'{cell}: starting new initial run in {out}', flush=True)
                result = subprocess.run(args, cwd=workspace)
                if result.returncode:
                    raise RuntimeError(f'Initial training failed ({result.returncode}); preserved {out}')
                continue
            step, source, sha, mode = found
            print(f'{cell}: verified {step}, checkpoint={source}, sha256={sha}', flush=True)
            if step >= target:
                break
            end = min((step // 10000 + 1) * 10000, target)
            out = root / cell / f'segment-{end:06d}-{uuid.uuid4().hex[:10]}'
            args = [sys.executable, '-u', '-m', 'exposure.mosaic_matched_cell',
                    '--complexity', CELLS[cell][0], '--imbalance', CELLS[cell][1],
                    '--trainer', 'warmstart', mode, '--source', str(source),
                    '--dataset', str(workspace / 'outputs' / 'mosaic-data-gpu-feasibility-81300'),
                    '--output', str(out), '--steps', str(end - step), '--device', 'cuda',
                    '--snapshot-steps', str(end)]
            if mode == 'warmstart':
                args.extend(['--seed', '81571'])
            process = subprocess.Popen(args, cwd=workspace)
            while process.poll() is None:
                if marker.exists():
                    subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'], check=True)
                    process.wait(timeout=30)
                    return
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f'Training failed ({process.returncode}); preserved {out}')
            completed, _, _ = from_report(out, cell)
            if completed != end:
                raise ValueError('Completed segment has wrong final step')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--cells', choices=CELLS, nargs='+', required=True)
    parser.add_argument('--target', type=int, default=500000)
    parser.add_argument('--wait-task')
    args = parser.parse_args()
    if os.name != 'nt' or len(set(args.cells)) != len(args.cells) or args.target < 150000:
        parser.error('Requires Windows, distinct cells and target >= 150000')
    # Protect against both scheduled and manually launched supervisors. The
    # process scan also protects against an orphaned training child.
    import msvcrt
    import ctypes
    root = args.workspace.resolve(strict=True) / 'outputs' / 'mosaic-distributed-81570'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'supervisor.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another distributed supervisor is running') from exc
        try:
            # Avoid idle sleep while active; explicit/lid-close sleep can still
            # occur and the process should continue after wake.
            if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
                raise OSError('Could not request prevention of idle sleep')
            supervise(args.workspace, args.cells, args.target, args.wait_task)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
