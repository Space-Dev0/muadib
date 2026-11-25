"""Audit fixed-plan worker allocation and replayed PNG bytes.

This is a descriptive audit of completed files. It does not evaluate image
semantics, prove device causation, or impute any missing result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from ops import adaptive_inference


def _read_json(path: Path) -> object:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def _identifier(value: object, label: str) -> str:
    return adaptive_inference._identifier(value, label)


def audit(manifest_path: Path, runs_root: Path) -> dict:
    """Audit the planner's primary assignments and distinct replay runs.

    Plan files live beside manifest at {base,repeat,cross}/<worker>-plan.json;
    independently executed coordinator runs live under runs_root/<phase>/<worker>.
    A run directory can be absent (not yet executed); a present run must contain
    the exact corresponding frozen plan. Every present image is hash/PNG checked.
    """
    manifest_path, runs_root = Path(manifest_path), Path(runs_root)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('invalid manifest schema')
    assignments, pairs = manifest.get('assignments'), manifest.get('audit_pairs')
    if not isinstance(assignments, list) or not assignments or not isinstance(pairs, list):
        raise ValueError('manifest requires assignments and audit_pairs lists')
    base_plan = _read_json(manifest_path.parent / 'base-plan.json')
    base_jobs = adaptive_inference.validate_plan(base_plan)
    canonical = json.dumps(base_plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if manifest.get('base_plan_sha256') != hashlib.sha256(canonical).hexdigest():
        raise ValueError('base plan digest does not match manifest')
    originals = {job['id']: job for job in base_jobs}

    assigned: dict[str, dict] = {}
    workers = set()
    for item in assignments:
        if not isinstance(item, dict) or not {'job_id', 'stratum', 'worker_id'} <= item.keys():
            raise ValueError('invalid primary assignment')
        jid = _identifier(item['job_id'], 'job id')
        _identifier(item['stratum'], 'stratum')
        worker = _identifier(item['worker_id'], 'worker id')
        if jid in assigned:
            raise ValueError('duplicate primary assignment')
        assigned[jid] = item
        workers.add(worker)
    if set(assigned) != set(originals):
        raise ValueError('primary assignments do not match the frozen base plan')

    selected = {}
    for pair in pairs:
        if not isinstance(pair, dict) or not {'job_id', 'stratum', 'primary_worker', 'repeat_worker', 'comparison_workers'} <= pair.keys():
            raise ValueError('invalid audit pair')
        jid = _identifier(pair['job_id'], 'job id')
        if jid not in assigned or jid in selected:
            raise ValueError('unknown or repeated replay job')
        p = assigned[jid]
        if pair['stratum'] != p['stratum'] or pair['primary_worker'] != p['worker_id'] or pair['repeat_worker'] != p['worker_id']:
            raise ValueError('audit pair does not match its primary assignment')
        comp = pair['comparison_workers']
        if not isinstance(comp, list) or len(set(comp)) != len(comp) or p['worker_id'] in comp:
            raise ValueError('invalid comparison workers')
        for w in comp:
            _identifier(w, 'comparison worker id')
        selected[jid] = pair
        workers.update(comp)

    # Fixed coordinator plans cannot contain an empty jobs list. Only require
    # plans for phases/workers with explicitly planned jobs.
    expected: dict[tuple[str, str], set[str]] = defaultdict(set)
    for jid, assignment in assigned.items():
        expected['base', assignment['worker_id']].add(jid)
    for jid, pair in selected.items():
        expected['repeat', pair['repeat_worker']].add(jid)
        for w in pair['comparison_workers']:
            expected['cross', w].add(jid)

    job_definitions: dict[str, dict] = {}
    plans: dict[tuple[str, str], dict] = {}
    for phase_worker, ids in sorted(expected.items()):
        phase, worker = phase_worker
        file = manifest_path.parent / phase / f'{worker}-plan.json'
        plan = _read_json(file)
        jobs = adaptive_inference.validate_plan(plan)
        mapped = {j['id']: j for j in jobs}
        if set(mapped) != ids:
            raise ValueError(f'{phase}/{worker} plan does not match manifest')
        plans[phase_worker] = plan
        if phase == 'base':
            for jid, job in mapped.items():
                if job['stratum'] != assigned[jid]['stratum']:
                    raise ValueError('job stratum does not match manifest')
                if job != originals[jid]:
                    raise ValueError('worker base plan differs from frozen base plan')
                job_definitions[jid] = job
    for (phase, worker), plan in plans.items():
        if phase != 'base':
            for job in plan['jobs']:
                if job != job_definitions[job['id']]:
                    raise ValueError(f'{phase}/{worker} replay differs from original job')

    records: dict[tuple[str, str, str], tuple[dict, bytes] | None] = {}
    for (phase, worker), plan in plans.items():
        run = runs_root / phase / worker
        if run.is_symlink():
            raise ValueError('run directory must not be a symlink')
        if run.exists():
            if not run.is_dir() or _read_json(run / 'plan.json') != plan:
                raise ValueError(f'{phase}/{worker} run plan differs from planner plan')
        for job in plan['jobs']:
            record = adaptive_inference._load_valid_result(run, job) if run.exists() else None
            if record is not None:
                # _load_valid_result verifies content, bounds and exact job metadata.
                actual = _identifier(record.get('worker_id'), 'record worker id')
                data = (run / 'results' / job['id'] / 'image.png').read_bytes()
                records[phase, worker, job['id']] = (record, data)
            else:
                records[phase, worker, job['id']] = None

    by_stratum = defaultdict(lambda: {'planned_primary': 0, 'completed_primary': 0,
                                      'missing_primary': 0, 'worker_assignment_mismatches': 0,
                                      'planned_by_worker': Counter(), 'completed_by_worker': Counter()})
    for jid, assignment in assigned.items():
        stratum, expected_worker = assignment['stratum'], assignment['worker_id']
        group = by_stratum[stratum]
        group['planned_primary'] += 1
        group['planned_by_worker'][expected_worker] += 1
        result = records['base', expected_worker, jid]
        if result is None:
            group['missing_primary'] += 1
        else:
            rec, _ = result
            group['completed_primary'] += 1
            group['completed_by_worker'][rec['worker_id']] += 1
            group['worker_assignment_mismatches'] += int(rec['worker_id'] != expected_worker)

    comparisons = defaultdict(Counter)
    def compare(kind: str, jid: str, phase: str, worker: str) -> None:
        original = assigned[jid]
        tally = comparisons[(original['stratum'], kind)]
        tally['planned_pairs'] += 1
        primary = records['base', original['worker_id'], jid]
        replay = records[phase, worker, jid]
        if primary is None:
            tally['missing_primary'] += 1
        if replay is None:
            tally['missing_replay'] += 1
        if primary is None or replay is None:
            tally['incomplete_pairs'] += 1
            return
        if primary[0]['worker_id'] != original['worker_id'] or replay[0]['worker_id'] != worker:
            tally['worker_assignment_mismatches'] += 1
            return
        tally['complete_pairs'] += 1
        tally['identical_png_bytes' if primary[1] == replay[1] else 'different_png_bytes'] += 1
    for jid, pair in selected.items():
        compare('same_worker_repeat', jid, 'repeat', pair['repeat_worker'])
        for worker in pair['comparison_workers']:
            compare('cross_worker_replay', jid, 'cross', worker)

    strata = {}
    for name, group in sorted(by_stratum.items()):
        group['planned_by_worker'] = dict(sorted(group['planned_by_worker'].items()))
        group['completed_by_worker'] = dict(sorted(group['completed_by_worker'].items()))
        strata[name] = group
    result_pairs = []
    keys = ('planned_pairs', 'complete_pairs', 'identical_png_bytes', 'different_png_bytes',
            'incomplete_pairs', 'missing_primary', 'missing_replay', 'worker_assignment_mismatches')
    for (stratum, kind), counts in sorted(comparisons.items()):
        result_pairs.append({'stratum': stratum, 'kind': kind, **{k: counts[k] for k in keys}})
    return {'schema_version': 1, 'strata': strata, 'comparisons': result_pairs,
            'notes': ['Primary denominators exclude all repeat and cross-worker replay jobs.',
                      'PNG byte equality is not a test of visual or semantic equivalence.',
                      'Worker identifiers are worker-reported labels, not hardware attestation.',
                      'Identical seeds do not ensure identical random streams across runtimes or devices.',
                      'Missing results remain missing; no image quality or causal effect is inferred.']}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(args.manifest, args.runs_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Saved descriptive audit to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
