"""Summarize completed scheduler sessions without concealing failed requests.

Usage: python -m ops.benchmark_report RUN_DIRECTORY [RUN_DIRECTORY ...]
This reports application request durations, not network one-way latency or GPU idle.
Completion counts are recorded scheduler values unless --verify-results is used.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def report_run(path: Path, *, verify_results: bool = False) -> dict:
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    plan = json.loads((path / 'plan.json').read_text(encoding='utf-8'))
    for field in ('planned', 'completed', 'missing', 'reused_jobs'):
        value = summary.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f'{field} must be a nonnegative integer')
    if summary['completed'] > summary['planned'] or summary['reused_jobs'] > summary['completed']:
        raise ValueError('Completion or reuse counts exceed their bounds')
    jobs = plan['jobs']
    job_ids = [j['id'] for j in jobs]
    if len(set(job_ids)) != len(job_ids) or len(jobs) != summary['planned']:
        raise ValueError('Plan must contain the reported number of unique jobs')
    if summary['completed'] + summary['missing'] != summary['planned']:
        raise ValueError('Inconsistent summary counts')
    if any(not isinstance(j.get('stratum'), str) or not j['stratum'] for j in jobs):
        raise ValueError('Every planned job must have a nonempty stratum')
    planned_strata = Counter(j['stratum'] for j in jobs)
    strata = summary.get('strata')
    if not isinstance(strata, dict) or set(strata) != set(planned_strata):
        raise ValueError('Summary strata do not match the planned strata')
    for name, expected in planned_strata.items():
        counts = strata[name]
        if not isinstance(counts, dict):
            raise ValueError(f'Invalid counts for stratum {name}')
        for field in ('planned', 'completed', 'missing'):
            if type(counts.get(field)) is not int or counts[field] < 0:
                raise ValueError(f'{name}.{field} must be a nonnegative integer')
        if counts['planned'] != expected or counts['completed'] + counts['missing'] != expected:
            raise ValueError(f'Inconsistent counts for stratum {name}')
    if any(sum(counts[field] for counts in strata.values()) != summary[field]
           for field in ('planned', 'completed', 'missing')):
        raise ValueError('Stratum totals do not match summary totals')
    if verify_results:
        # Reuse the coordinator's plan, hash, size, PNG and filesystem checks.
        from ops import adaptive_inference
        adaptive_inference.validate_plan(plan)
        verified = Counter()
        for job in jobs:
            if adaptive_inference._load_valid_result(path, job) is not None:
                verified[job['stratum']] += 1
        if any(verified[name] != counts['completed'] for name, counts in strata.items()):
            raise ValueError('Verified result files do not match recorded completion counts')
    events = [json.loads(line) for line in (path / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    # Record settings per session, rather than assuming a resumed segment used
    # the same policy or workers. Older runs may not contain these fields.
    session_events = [e for e in events if e.get('event') == 'run_started']
    session_configs = {e.get('session_id'): e.get('session_config') for e in session_events}
    current_config = session_configs.get(summary['session_id'])
    if current_config is not None and not isinstance(current_config, dict):
        raise ValueError('Invalid current session configuration')
    ignored_config_keys = {'resume', 'allow_uncertain_retry'}
    changes = {}
    missing_config = []
    for sid, config in session_configs.items():
        if sid == summary['session_id']:
            continue
        if config is None:
            missing_config.append(sid)
        elif not isinstance(config, dict):
            raise ValueError('Invalid previous session configuration')
        elif current_config is not None:
            differences = sorted(k for k in (set(config) | set(current_config)) - ignored_config_keys
                                 if config.get(k) != current_config.get(k))
            if differences:
                changes[sid] = differences
    timing = {}
    for field in ('setup_seconds', 'result_validation_seconds', 'session_elapsed_before_summary_seconds'):
        value = summary.get(field)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(f'Invalid {field}')
        timing[field] = value
    all_starts = [e for e in events if e.get('event') == 'attempt_started']
    starts_by_id = {e['attempt_id']: e for e in all_starts}
    if len(starts_by_id) != len(all_starts):
        raise ValueError('Duplicate historical attempt ID')
    # Legacy recovery events omit session_id. Their original start identifies the
    # session in which the attempt became uncertain, not the session that resumed it.
    def event_session(event):
        return event.get('session_id') or starts_by_id.get(event.get('attempt_id'), {}).get('session_id')

    current = [e for e in events if event_session(e) == summary['session_id']]
    succeeded = [e for e in current if e.get('event') == 'attempt_succeeded']
    failed = [e for e in current if e.get('event') == 'attempt_failed']
    starts = [e for e in current if e.get('event') == 'attempt_started']
    attempt_ids = [e['attempt_id'] for e in starts]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError('Duplicate attempt ID')
    abandoned = [e for e in current if e.get('event') == 'attempt_abandoned']
    endings = succeeded + failed + abandoned
    ending_ids = [e['attempt_id'] for e in endings]
    if len(set(ending_ids)) != len(ending_ids) or not set(ending_ids) <= set(attempt_ids):
        raise ValueError('Invalid attempt end events')
    succeeded_ids = [e['job_id'] for e in succeeded]
    if len(set(succeeded_ids)) != len(succeeded_ids) or not set(succeeded_ids) <= set(job_ids):
        raise ValueError('Duplicate or unknown successful job')
    if len(succeeded) != summary['completed'] - summary['reused_jobs']:
        raise ValueError('Session successes do not match the completion summary')
    # duration_seconds also includes coordinator processing delays. Never mix it
    # with the timer measured inside request(), even for legacy events.
    durations = [float(e['roundtrip_seconds']) for e in succeeded if 'roundtrip_seconds' in e]
    if any(not math.isfinite(v) or v < 0 for v in durations):
        raise ValueError('Invalid success duration')
    wall = float(summary['segment_wall_seconds'])
    if not math.isfinite(wall) or wall < 0:
        raise ValueError('Invalid segment duration')
    all_abandoned = [e for e in events if e.get('event') == 'attempt_abandoned']
    uncertain_failed = [e for e in events if e.get('event') == 'attempt_failed' and e.get('uncertain') is True]
    all_end_ids = {e['attempt_id'] for e in events if e.get('event') in
                   ('attempt_succeeded', 'attempt_failed', 'attempt_abandoned')}
    unresolved_starts = set(starts_by_id) - all_end_ids
    orphan_abandoned = sum(e.get('attempt_id') not in starts_by_id for e in all_abandoned)
    return {
        'run': str(path), 'session_id': summary['session_id'], 'segment_policy': summary['policy'],
        'planned': summary['planned'], 'recorded_completed_total': summary['completed'],
        'missing': summary['missing'], 'reused_from_previous_sessions': summary['reused_jobs'],
        'successes_this_session': len(succeeded), 'failed_attempts_this_session': len(failed),
        'abandoned_attempts_this_session': len(abandoned),
        'uncertain_failed_attempts_this_session': sum(e.get('uncertain') is True for e in failed),
        'unfinished_attempts_this_session': len(starts) - len(endings),
        'historical_attempts': {'abandoned': len(all_abandoned),
                                'uncertain_failed': len(uncertain_failed),
                                'starts_without_recorded_outcome': len(unresolved_starts),
                                'abandoned_without_matching_start': orphan_abandoned},
        'successes_by_worker': dict(Counter(e['worker_id'] for e in succeeded)),
        'success_request_duration_seconds': {
            'n': len(durations), 'missing_legacy_measurement': len(succeeded) - len(durations),
            'median': statistics.median(durations) if durations else None,
            'median_method': 'arithmetic median; average middle values when n is even',
            'p95': _quantile(durations, .95),
            'p95_method': 'linear interpolation at zero-based rank (n - 1) * 0.95'},
        'segment_wall_seconds': wall,
        'session_timing_seconds': timing,
        'session_configuration': current_config,
        'prior_session_configuration_changes': changes,
        'prior_sessions_without_configuration': missing_config,
        'new_successes_per_segment_second': len(succeeded) / wall if wall > 0 else None,
        'strata': strata,
        'result_artifacts_independently_validated': verify_results,
        'notes': [('Result files, PNG structure and hashes were independently checked against the plan.'
                   if verify_results else 'Completion counts reflect recorded scheduler events and summary; result files and hashes are not independently checked.'),
                  'Request duration is a coordinator monotonic interval from request() entry through connection, remote sampling, response transfer, SHA-256 and structural PNG verification; success-only, with legacy missing values excluded.',
                  'Throughput is new completions in this session divided by dispatch segment time; resumed work is excluded.',
                  'Segment time excludes resume validation and setup, and is not end-to-end elapsed time.',
                  'Setup includes result validation; elapsed-before-summary covers setup and dispatch but excludes summary serialization and writing.',
                  'The policy applies only to the current segment; reused jobs may have been completed under another policy.',
                  'Abandoned and uncertain failed attempts may still be running remotely, including after a retry succeeds.',
                  'p95 uses linear interpolation; the median averages the middle values for even samples. Small samples do not establish a reliable tail.',
                  'Request duration excludes queueing and result publication after request() returns; it is not one-way network latency, jitter, or GPU idle time.'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--verify-results', action='store_true',
                        help='verify plan, result files, image format and hashes against completion counts')
    args = parser.parse_args()
    try:
        print(json.dumps([report_run(path, verify_results=args.verify_results) for path in args.runs], indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
