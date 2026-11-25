"""Construct a pinned, stratified baseline and separate cross-device audits.

The resulting independent coordinator plans do not implement automatic
failover. Run each worker and phase separately, and preserve incomplete work.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from ops import adaptive_inference


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_design(plan: dict, workers: list[dict], *, audit_per_stratum: int = 1,
                  selection_seed: int = 0) -> dict:
    """Make deterministic assignments and independent replica-phase plans.

    Assignment greedily balances stratum and global counts among compatible
    workers; it is not a global optimum or a causal randomization design.
    Audit selection depends only on declared input, never on observed images.
    """
    jobs = adaptive_inference.validate_plan(plan)
    workers = adaptive_inference.validate_workers(workers)
    if type(audit_per_stratum) is not int or audit_per_stratum < 0:
        raise ValueError('audit_per_stratum must be a nonnegative integer')
    if type(selection_seed) is not int or not 0 <= selection_seed < 2**63:
        raise ValueError('selection_seed must be an integer in [0, 2**63)')
    if len({w['id'].casefold() for w in workers}) != len(workers):
        raise ValueError('worker IDs must differ even on case-insensitive filesystems')
    ordered_workers = sorted(workers, key=lambda w: w['id'].casefold())
    compatible = {j['id']: [w for w in ordered_workers
                             if w['checkpoint_sha256'] == j['checkpoint_sha256']] for j in jobs}
    if any(not compatible[j['id']] for j in jobs):
        raise ValueError('every job requires at least one checkpoint-compatible worker')

    phases = {name: defaultdict(list) for name in ('base', 'repeat', 'cross')}
    counts = defaultdict(Counter)
    totals = Counter()
    assignments = []
    primary = {}
    # Stable input-independent order avoids dependence on JSON list order.
    for job in sorted(jobs, key=lambda j: (j['stratum'], j['id'].casefold())):
        worker = min(compatible[job['id']], key=lambda w: (
            counts[job['stratum']][w['id']], totals[w['id']], w['id'].casefold()))
        wid = worker['id']
        counts[job['stratum']][wid] += 1
        totals[wid] += 1
        primary[job['id']] = wid
        assignments.append({'job_id': job['id'], 'stratum': job['stratum'], 'worker_id': wid})
        phases['base'][wid].append(dict(job))

    groups = defaultdict(list)
    for job in jobs:
        groups[job['stratum']].append(job)
    pairs = []
    for stratum in sorted(groups):
        def rank(job):
            value = {'selection_seed': selection_seed, 'stratum': stratum, 'job_id': job['id']}
            return (_digest(value), job['id'].casefold())
        for job in sorted(groups[stratum], key=rank)[:audit_per_stratum]:
            wid = primary[job['id']]
            alternatives = [w['id'] for w in compatible[job['id']] if w['id'] != wid]
            phases['repeat'][wid].append(dict(job))
            for alternate in alternatives:
                phases['cross'][alternate].append(dict(job))
            pairs.append({'job_id': job['id'], 'stratum': stratum, 'primary_worker': wid,
                          'repeat_worker': wid, 'comparison_workers': alternatives})

    # Validate every emitted plan against the actual coordinator's constraints.
    plan_files = {}
    for phase, by_worker in phases.items():
        for wid, group in by_worker.items():
            subplan = {'schema_version': 1, 'jobs': group}
            adaptive_inference.validate_plan(subplan)
            plan_files[f'{phase}/{wid}-plan.json'] = subplan
    worker_files = {f'workers/{w["id"]}.json': {'workers': [dict(w)]} for w in ordered_workers}
    manifest = {'schema_version': 1, 'algorithm': 'greedy-stratum-then-total-v1',
                'base_plan_sha256': _digest(plan), 'workers_sha256': _digest(workers),
                'selection_seed': selection_seed, 'audit_per_stratum': audit_per_stratum,
                'assignments': assignments, 'audit_pairs': pairs,
                'plan_files': sorted(plan_files), 'worker_files': sorted(worker_files),
                'limitations': ['Pinned assignments do not automatically fail over.',
                    'Replicas are audit observations and must not count as original samples.',
                    'Matching seeds across devices does not guarantee matching random streams or images.',
                    'Balance is greedy and constrained by checkpoint compatibility.']}
    return {'manifest': manifest, 'plans': plan_files, 'workers': worker_files,
            'base_plan': plan, 'base_assignments': {'schema_version': 1, 'assignments': primary}}


def write_design(output: Path, design: dict) -> None:
    """Create a new design directory without overwriting previous evidence."""
    output.mkdir(parents=True, exist_ok=False)
    for name, value in {**design['plans'], **design['workers'], 'manifest.json': design['manifest'],
                        'base-plan.json': design['base_plan'], 'base-assignments.json': design['base_assignments']}.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--workers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-per-stratum', type=int, default=1)
    parser.add_argument('--selection-seed', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
        config = json.loads(args.workers.read_text(encoding='utf-8'))
        if not isinstance(config, dict) or set(config) != {'workers'}:
            raise ValueError('workers file must contain only workers')
        design = create_design(plan, config['workers'], audit_per_stratum=args.audit_per_stratum,
                               selection_seed=args.selection_seed)
        write_design(args.output, design)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f"Wrote {len(design['manifest']['assignments'])} base assignments and "
          f"{len(design['manifest']['audit_pairs'])} audit pairs to {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
