"""Supervise bounded PilotUNet segments until a persistent stop marker exists.

Windows scheduled-task wrapper; preserves incomplete output directories.
"""
import argparse
import ctypes
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
from exposure.mosaic_pilotunet_warmstart import load_warmstart_checkpoint

MANIFEST = '48bf70c4d43c07bdc563cba1f585ea618e862047635257a4ce63f67c9bd45d37'
INITIAL_HASH = 'b3405f731ff44963fb1704687108e90499812bb21657410cbd78dab051ecca8c'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_checkpoint(path, step, expected_hash=None):
    """Validate file, format and known experiment config; hash is checked if known."""
    path = Path(path).resolve(strict=True)
    actual_hash = digest(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError('Checkpoint hash mismatch')
    checkpoint = load_warmstart_checkpoint(path)
    config = checkpoint['config']
    expected = {'width': 16, 'timesteps': 1000, 'batch_size': 4, 'seed': 81377,
                'lr': 0.0001, 'dropout': 0.1, 'ema_decay': 0.999, 'deterministic': False}
    if (checkpoint.get('cell') != ['low', 'balanced'] or
        checkpoint['report'].get('cumulative_step') != step or
        checkpoint['provenance']['dataset'].get('manifest_sha256') != MANIFEST or
        any(config.get(k) != v for k, v in expected.items())):
        raise ValueError('Checkpoint experiment configuration mismatch')
    if (checkpoint['rng'].get('device') != 'cuda' or not torch.cuda.is_available()
            or not isinstance(checkpoint['rng'].get('torch_cuda'), list)
            or len(checkpoint['rng']['torch_cuda']) != torch.cuda.device_count()
            or checkpoint['provenance'].get('torch') != str(torch.__version__)
            or checkpoint['provenance'].get('python') != sys.version):
        raise ValueError('Checkpoint CUDA or runtime does not match')
    recorded = checkpoint['provenance']['runtime']['code_sha256']
    for name in ('pilot_train', 'mosaic_pilotunet_probe', 'mosaic_pilotunet_warmstart',
                 'mosaic_toy_train', 'velocity_probe', 'identity_eps_finetune',
                 'measure', 'render', 'rules'):
        module = importlib.import_module(f'exposure.{name}')
        if recorded.get(name) != digest(Path(module.__file__)):
            raise ValueError(f'Checkpoint source mismatch: {name}')
    return actual_hash


def completed_report_checkpoint(directory, step):
    """Require report snapshot hash and loadable checkpoint for a finished segment."""
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    matches = [x for x in report['snapshots'] if x['step'] == step]
    if len(matches) != 1:
        raise ValueError('Missing or duplicate final snapshot in report')
    path = directory / f'warmstart-{step:06d}.pt'
    sha = validate_checkpoint(path, step, matches[0]['checkpoint_sha256'])
    return path.resolve(), sha


def initial_checkpoint(workspace):
    """Prefer a completed predecessor report; otherwise select a loadable snapshot."""
    workspace = Path(workspace)
    predecessor = workspace / 'outputs' / 'mosaic-pilotunet-overnight-81390'
    report_path = predecessor / 'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        steps = sorted((x['step'] for x in report['snapshots']), reverse=True)
        if not steps:
            raise ValueError('Predecessor report has no snapshots')
        path, sha = completed_report_checkpoint(predecessor, steps[0])
        return steps[0], path, sha, 'report_hash'
    # An interrupted predecessor has no final report. Loadability/schema/config
    # checks cannot provide a known-good external hash for these files.
    candidates = sorted(predecessor.glob('warmstart-*.pt'), reverse=True)
    for path in candidates:
        try:
            step = int(path.stem.removeprefix('warmstart-'))
            sha = validate_checkpoint(path, step)
            return step, path.resolve(), sha, 'loadable_without_report_hash'
        except Exception as exc:
            print(f'Skipping unreadable predecessor checkpoint {path}: {exc}', flush=True)
            continue
    path = workspace / 'outputs' / 'mosaic-pilotunet-resume-81379' / 'warmstart-030000.pt'
    return 30000, path.resolve(), validate_checkpoint(path, 30000, INITIAL_HASH), 'known_initial_hash'


def write_state(path, state):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as file:
        json.dump(state, file, indent=2, allow_nan=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def predecessor_running():
    command = ['powershell.exe', '-NoProfile', '-Command',
               "(Get-ScheduledTask -TaskName 'MuadibPilotUNetOvernight81390' -ErrorAction Stop).State"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    state = result.stdout.strip()
    if state not in ('Running', 'Ready', 'Disabled', 'Queued'):
        raise RuntimeError(f'Unexpected predecessor task state: {state!r}')
    return state in ('Running', 'Queued')


def supervise(workspace, control, *, interval=10000):
    workspace, control = Path(workspace).resolve(strict=True), Path(control)
    control.mkdir(parents=True, exist_ok=True)
    marker, state_path = control / 'STOP', control / 'state.json'
    output_root = workspace / 'outputs' / 'mosaic-pilotunet-continuous-81391'
    output_root.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        print('Stop marker exists; no training started.', flush=True)
        return
    while predecessor_running():
        if marker.exists():
            print('Stop requested while waiting for predecessor.', flush=True)
            return
        time.sleep(10)
    if marker.exists():
        return
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        step, source = state['step'], Path(state['checkpoint']).resolve(strict=True)
        if not source.is_relative_to(workspace):
            raise ValueError('State checkpoint is outside workspace')
        sha = validate_checkpoint(source, step, state['sha256'])
        verification = 'saved_state_hash'
    else:
        step, source, sha, verification = initial_checkpoint(workspace)
    while not marker.exists():
        write_state(state_path, {'step': step, 'checkpoint': str(source), 'sha256': sha,
                                 'verification': verification, 'status': 'ready'})
        next_step = step + interval
        target = output_root / f'segment-{next_step:06d}-{uuid.uuid4().hex[:10]}'
        args = [sys.executable, '-u', '-m', 'exposure.mosaic_pilotunet_warmstart', 'resume',
                '--source', str(source), '--dataset', str(workspace / 'outputs' / 'mosaic-data-gpu-feasibility-81300'),
                '--output', str(target), '--steps', str(interval), '--device', 'cuda',
                '--snapshot-steps', str(next_step)]
        print(f'Starting {step} -> {next_step}; source={source}; verification={verification}', flush=True)
        process = subprocess.Popen(args, cwd=workspace)
        while process.poll() is None:
            if marker.exists():
                print('Stop requested; terminating current segment. Partial output is preserved.', flush=True)
                # The Windows venv launcher may spawn the actual Python process.
                # Terminate the owned process tree, not only the launcher PID.
                result = subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                                        capture_output=True, text=True, timeout=30)
                if result.returncode != 0 and process.poll() is None:
                    raise RuntimeError(f'Could not stop training process tree: {result.stderr}')
                process.wait(timeout=30)
                return
            time.sleep(2)
        if process.returncode != 0:
            raise RuntimeError(f'Segment failed with exit code {process.returncode}; outputs preserved at {target}')
        source, sha = completed_report_checkpoint(target, next_step)
        step, verification = next_step, 'report_hash'
        write_state(state_path, {'step': step, 'checkpoint': str(source), 'sha256': sha,
                                 'verification': verification, 'status': 'ready'})
    print('Stop requested; exiting.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('Supervisor requires Windows')
    from ctypes import wintypes
    kernel = ctypes.windll.kernel32
    kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.SetThreadExecutionState.argtypes = (wintypes.DWORD,)
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    handle = kernel.CreateMutexW(None, False, 'Local\\MuadibPilotUNetContinuous81391')
    if not handle:
        raise OSError('Cannot create training supervisor mutex')
    acquired = kernel.WaitForSingleObject(handle, 0)
    if acquired not in (0, 0x80):
        kernel.CloseHandle(handle)
        raise RuntimeError('Another continuous supervisor is already running')
    previous = kernel.SetThreadExecutionState(0x80000001)
    if not previous:
        kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)
        raise OSError('Cannot request idle-sleep prevention')
    try:
        supervise(args.workspace, args.control)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    main()
