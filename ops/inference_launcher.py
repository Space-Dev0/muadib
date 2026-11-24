"""Launch SSH forwards for an already-running fixed-plan inference job.

Only tunnels started by this process are stopped. SSH uses existing host-key
verification and noninteractive credentials. Remote worker lifecycle is external.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import warnings

from ops import adaptive_inference, protocol

_HOST = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]*(?:@[A-Za-z0-9_][A-Za-z0-9_.-]*)?$')


def validate_tunnels(config: object, workers: list[dict]) -> list[tuple[dict, dict]]:
    if not isinstance(config, dict) or set(config) != {'tunnels'} or not isinstance(config['tunnels'], list):
        raise ValueError('tunnel configuration requires a tunnels list')
    by_id = {w['id']: w for w in adaptive_inference.validate_workers(workers)}
    result = []
    seen = set()
    for tunnel in config['tunnels']:
        if (not isinstance(tunnel, dict)
                or not {'worker_id', 'host', 'ssh_port', 'remote_port'} <= set(tunnel)
                or set(tunnel) - {'worker_id', 'host', 'ssh_port', 'remote_port', 'control_path'}):
            raise ValueError('tunnel requires worker_id, host, ssh_port and remote_port; control_path is optional')
        identity = tunnel['worker_id']
        host = tunnel['host']
        if not isinstance(identity, str) or identity not in by_id or identity in seen:
            raise ValueError('unknown or duplicate tunnel worker_id')
        if not isinstance(host, str) or not _HOST.fullmatch(host):
            raise ValueError('host must be an SSH hostname or user@hostname')
        for name in ('ssh_port', 'remote_port'):
            if type(tunnel[name]) is not int or not 1 <= tunnel[name] <= 65535:
                raise ValueError(f'{name} must be between 1 and 65535')
        if 'control_path' in tunnel:
            path = tunnel['control_path']
            if not isinstance(path, str) or not path or '\x00' in path:
                raise ValueError('control_path must be an absolute path to an existing socket')
            control = Path(path)
            if not control.is_absolute() or control.is_symlink() or not control.is_socket():
                raise ValueError('control_path must be an absolute non-symlink existing socket')
        seen.add(identity)
        result.append((tunnel, by_id[identity]))
    if seen != set(by_id):
        raise ValueError('each worker must have exactly one tunnel')
    return result


def _available(port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError):
        return False


def _worker_ready(port: int, deadline: float) -> bool:
    # An invalid request causes the legacy worker to return an error without
    # starting GPU sampling. This proves protocol responsiveness, not identity.
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=min(0.5, protocol.remaining(deadline))) as sock:
            sock.settimeout(protocol.remaining(deadline))
            protocol.send_json(sock, {})
            response = protocol.recv_json(sock, deadline=deadline)
            protocol.remaining(deadline)
            return isinstance(response, dict) and response.get('status') == 'error'
    except (OSError, EOFError, ValueError, TypeError):
        return False


def _cancel_forward(cancel_args: list[str]) -> None:
    try:
        result = subprocess.run(cancel_args, stdin=subprocess.DEVNULL, timeout=5, check=False)
        if result.returncode != 0:
            warnings.warn('Could not confirm cancellation of launcher-owned multiplexed forward', RuntimeWarning)
    except (OSError, subprocess.TimeoutExpired) as exc:
        warnings.warn(f'Could not cancel launcher-owned multiplexed forward: {exc}', RuntimeWarning)


def _stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def launch(plan: dict, workers: list[dict], config: dict, output: Path, *, policy: str = 'adaptive',
           timeout: float = 600, startup_timeout: float = 15, resume: bool = False,
           allow_uncertain_retry: bool = False) -> dict:
    adaptive_inference.validate_plan(plan)
    tunnels = validate_tunnels(config, workers)
    if not 0 < startup_timeout <= 300:
        raise ValueError('startup_timeout must be in (0, 300]')
    # Fail early when an existing listener would make readiness ambiguous.
    for _, worker in tunnels:
        if _available(worker['port']):
            raise RuntimeError(f"local port {worker['port']} is already accepting connections")
    with ExitStack() as stack:
        processes = []
        for tunnel, worker in tunnels:
            forward = f"127.0.0.1:{worker['port']}:127.0.0.1:{tunnel['remote_port']}"
            process = None
            deadline = time.monotonic() + startup_timeout
            if 'control_path' in tunnel:
                base = ['ssh', '-S', tunnel['control_path'], '-O']
                suffix = ['-L', forward, '-p', str(tunnel['ssh_port']), tunnel['host']]
                try:
                    result = subprocess.run(base + ['forward'] + suffix, stdin=subprocess.DEVNULL,
                                            timeout=protocol.remaining(deadline), check=False)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"SSH forward for {worker['id']} timed out; inspect master forwards") from exc
                if result.returncode != 0:
                    raise RuntimeError(f"SSH forward for {worker['id']} failed")
                stack.callback(_cancel_forward, base + ['cancel'] + suffix)
            else:
                # Explicitly disable implicit multiplexing from ~/.ssh/config;
                # otherwise a pre-existing master could own and retain the forward.
                args = ['ssh', '-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                        '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
                        '-p', str(tunnel['ssh_port']), '-L', forward, tunnel['host']]
                process = subprocess.Popen(args, stdin=subprocess.DEVNULL)
                stack.callback(_stop_owned, process)
                processes.append(process)
            while True:
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"SSH tunnel for {worker['id']} exited before readiness")
                if _worker_ready(worker['port'], deadline):
                    if process is not None:
                        time.sleep(0.15)
                        if process.poll() is not None:
                            raise RuntimeError(f"SSH tunnel for {worker['id']} exited during startup")
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"SSH tunnel for {worker['id']} was not ready before deadline")
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"SSH tunnel for {worker['id']} was not ready before deadline")
                time.sleep(0.1)
        if any(process.poll() is not None for process in processes):
            raise RuntimeError('SSH tunnel exited')
        return adaptive_inference.run(plan, workers, output, policy=policy, timeout=timeout,
                                      resume=resume, allow_uncertain_retry=allow_uncertain_retry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('plan', 'workers', 'tunnels', 'output'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--policy', choices=('adaptive', 'round-robin'), default='adaptive')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--startup-timeout', type=float, default=15)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-uncertain-retry', action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
        config = json.loads(args.workers.read_text(encoding='utf-8'))
        if not isinstance(config, dict) or set(config) != {'workers'}:
            raise ValueError('worker configuration must contain only workers')
        tunnels = json.loads(args.tunnels.read_text(encoding='utf-8'))
        summary = launch(plan, config['workers'], tunnels, args.output, policy=args.policy,
                         timeout=args.timeout, startup_timeout=args.startup_timeout,
                         resume=args.resume, allow_uncertain_retry=args.allow_uncertain_retry)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['missing'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
