import json
import io
import tempfile
import unittest
from pathlib import Path

from ops.benchmark_report import report_run


class BenchmarkReportTests(unittest.TestCase):
    def test_failure_and_resume_are_not_counted_as_new_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'plan.json').write_text(json.dumps({'jobs': [{'id': 'a', 'stratum': 's'}, {'id': 'b', 'stratum': 's'}]}))
            (path / 'summary.json').write_text(json.dumps({'session_id': 's', 'policy': 'adaptive',
                'planned': 2, 'completed': 2, 'missing': 0, 'reused_jobs': 1,
                'segment_wall_seconds': 4, 'strata': {'s': {'planned': 2, 'completed': 2, 'missing': 0}}}))
            events = [
                {'event': 'attempt_started', 'session_id': 's', 'attempt_id': '1'},
                {'event': 'attempt_failed', 'session_id': 's', 'attempt_id': '1'},
                {'event': 'attempt_started', 'session_id': 's', 'attempt_id': '2'},
                {'event': 'attempt_succeeded', 'session_id': 's', 'attempt_id': '2',
                 'job_id': 'b', 'worker_id': 'gpu', 'duration_seconds': 2, 'roundtrip_seconds': 1.5},
                {'event': 'attempt_succeeded', 'session_id': 'old', 'attempt_id': '3',
                 'job_id': 'a', 'worker_id': 'gpu', 'duration_seconds': 50},
            ]
            (path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            result = report_run(path)
            self.assertFalse(result['result_artifacts_independently_validated'])
            self.assertEqual(result['recorded_completed_total'], 2)
            self.assertEqual(result['segment_policy'], 'adaptive')
            self.assertTrue(any('another policy' in note for note in result['notes']))
            self.assertEqual(result['failed_attempts_this_session'], 1)
            self.assertEqual(result['successes_this_session'], 1)
            self.assertEqual(result['new_successes_per_segment_second'], .25)
            self.assertEqual(result['success_request_duration_seconds']['median'], 1.5)
            self.assertIn('average middle values', result['success_request_duration_seconds']['median_method'])
            self.assertIn('linear interpolation', result['success_request_duration_seconds']['p95_method'])
            self.assertTrue(any('excludes queueing and result publication' in note for note in result['notes']))

    def test_historical_abandoned_legacy_and_uncertain_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'plan.json').write_text(json.dumps({'jobs': [{'id': 'a', 'stratum': 's'}]}))
            (path / 'summary.json').write_text(json.dumps({'session_id': 'new', 'policy': 'adaptive',
                'planned': 1, 'completed': 1, 'missing': 0, 'reused_jobs': 0,
                'segment_wall_seconds': 5, 'strata': {'s': {'planned': 1, 'completed': 1, 'missing': 0}}}))
            events = [
                {'event': 'attempt_started', 'session_id': 'old', 'attempt_id': 'a1'},
                {'event': 'attempt_abandoned', 'attempt_id': 'a1'},
                {'event': 'attempt_started', 'session_id': 'old', 'attempt_id': 'a2'},
                {'event': 'attempt_failed', 'session_id': 'old', 'attempt_id': 'a2', 'uncertain': True},
                {'event': 'attempt_started', 'session_id': 'old', 'attempt_id': 'a3'},
                {'event': 'attempt_started', 'session_id': 'new', 'attempt_id': 'a4'},
                {'event': 'attempt_succeeded', 'session_id': 'new', 'attempt_id': 'a4',
                 'job_id': 'a', 'worker_id': 'gpu', 'duration_seconds': 3},
            ]
            (path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            result = report_run(path)
            self.assertEqual(result['historical_attempts']['abandoned'], 1)
            self.assertEqual(result['historical_attempts']['uncertain_failed'], 1)
            self.assertEqual(result['historical_attempts']['starts_without_recorded_outcome'], 1)
            self.assertEqual(result['historical_attempts']['abandoned_without_matching_start'], 0)
            self.assertEqual(result['unfinished_attempts_this_session'], 0)
            self.assertTrue(any('excludes resume validation' in note for note in result['notes']))

    def test_missing_request_timing_is_excluded_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'plan.json').write_text(json.dumps({'jobs': [{'id': 'a', 'stratum': 's'}, {'id': 'b', 'stratum': 's'}]}))
            (path / 'summary.json').write_text(json.dumps({'session_id': 's', 'policy': 'adaptive',
                'planned': 2, 'completed': 2, 'missing': 0, 'reused_jobs': 0,
                'segment_wall_seconds': 10, 'strata': {'s': {'planned': 2, 'completed': 2, 'missing': 0}}}))
            events = [
                {'event': 'attempt_started', 'session_id': 's', 'attempt_id': '1'},
                {'event': 'attempt_succeeded', 'session_id': 's', 'attempt_id': '1',
                 'job_id': 'a', 'worker_id': 'gpu', 'duration_seconds': 7, 'roundtrip_seconds': 2},
                {'event': 'attempt_started', 'session_id': 's', 'attempt_id': '2'},
                {'event': 'attempt_succeeded', 'session_id': 's', 'attempt_id': '2',
                 'job_id': 'b', 'worker_id': 'gpu', 'duration_seconds': 8},
            ]
            (path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            result = report_run(path)['success_request_duration_seconds']
            self.assertEqual(result['n'], 1)
            self.assertEqual(result['missing_legacy_measurement'], 1)
            self.assertEqual(result['median'], 2)

    def test_invalid_summary_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'plan.json').write_text(json.dumps({'jobs': [{'id': 'a', 'stratum': 's'}, {'id': 'b', 'stratum': 's'}]}))
            (path / 'events.jsonl').write_text('')
            base = {'session_id': 's', 'policy': 'adaptive', 'planned': 2, 'completed': 0,
                    'missing': 2, 'reused_jobs': 0, 'segment_wall_seconds': 1, 'strata': {}}
            for changes in ({'completed': 3, 'missing': -1}, {'completed': 3, 'missing': 0},
                            {'reused_jobs': 1}, {'planned': True}, {'missing': 2.0}):
                with self.subTest(changes=changes):
                    (path / 'summary.json').write_text(json.dumps({**base, **changes}))
                    with self.assertRaises(ValueError):
                        report_run(path)

    def test_quantile_of_empty_run_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'plan.json').write_text(json.dumps({'jobs': [{'id': 'a', 'stratum': 's'}]}))
            (path / 'summary.json').write_text(json.dumps({'session_id': 's', 'policy': 'round_robin',
                'planned': 1, 'completed': 0, 'missing': 1, 'reused_jobs': 0,
                'segment_wall_seconds': 2, 'strata': {'s': {'planned': 1, 'completed': 0, 'missing': 1}}}))
            (path / 'events.jsonl').write_text('')
            self.assertIsNone(report_run(path)['success_request_duration_seconds']['p95'])

    def test_stratum_counts_must_match_plan_and_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            jobs = [{'id': 'a', 'stratum': 'low'}, {'id': 'b', 'stratum': 'high'}]
            (path / 'plan.json').write_text(json.dumps({'jobs': jobs}))
            (path / 'events.jsonl').write_text('')
            strata = {'low': {'planned': 1, 'completed': 0, 'missing': 1},
                      'high': {'planned': 1, 'completed': 0, 'missing': 1}}
            base = {'session_id': 's', 'policy': 'adaptive', 'planned': 2, 'completed': 0,
                    'missing': 2, 'reused_jobs': 0, 'segment_wall_seconds': 1, 'strata': strata}
            (path / 'summary.json').write_text(json.dumps(base))
            self.assertEqual(report_run(path)['strata'], strata)
            for bad in ({'low': {'planned': 2, 'completed': 0, 'missing': 2}},
                        {'low': {'planned': 1, 'completed': -1, 'missing': 2}, 'high': strata['high']},
                        {'low': {'planned': 1, 'completed': 1, 'missing': 0}, 'high': strata['high']}):
                with self.subTest(bad=bad):
                    (path / 'summary.json').write_text(json.dumps({**base, 'strata': bad}))
                    with self.assertRaises(ValueError):
                        report_run(path)

    def test_optional_artifact_verification_checks_hash_and_counts(self):
        from PIL import Image
        from ops import adaptive_inference
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            job = {'id': 'a', 'stratum': 's', 'checkpoint_sha256': 'a' * 64,
                   'color': 'red', 'seed': 1}
            (path / 'plan.json').write_text(json.dumps({'schema_version': 1, 'jobs': [job]}))
            (path / 'summary.json').write_text(json.dumps({'session_id': 's', 'policy': 'adaptive',
                'planned': 1, 'completed': 1, 'missing': 0, 'reused_jobs': 1,
                'segment_wall_seconds': 1, 'strata': {'s': {'planned': 1, 'completed': 1, 'missing': 0}}}))
            (path / 'events.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError, 'result files'):
                report_run(path, verify_results=True)
            (path / 'results').mkdir()
            buf = io.BytesIO()
            Image.new('RGB', (64, 64)).save(buf, format='PNG')
            adaptive_inference._publish(path, job, buf.getvalue(), {'id': 'w'}, 1, {}, 'attempt')
            self.assertTrue(report_run(path, verify_results=True)['result_artifacts_independently_validated'])
            (path / 'results' / 'a' / 'image.png').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                report_run(path, verify_results=True)

    def test_session_settings_changes_and_timing_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'plan.json').write_text(json.dumps({'jobs': [{'id': 'a', 'stratum': 's'}]}))
            (path / 'summary.json').write_text(json.dumps({'session_id': 'new', 'policy': 'adaptive',
                'planned': 1, 'completed': 0, 'missing': 1, 'reused_jobs': 0,
                'segment_wall_seconds': 2, 'setup_seconds': 1, 'result_validation_seconds': .4,
                'session_elapsed_before_summary_seconds': 3,
                'strata': {'s': {'planned': 1, 'completed': 0, 'missing': 1}}}))
            events = [
                {'event': 'run_started', 'session_id': 'legacy'},
                {'event': 'run_started', 'session_id': 'old', 'session_config':
                 {'policy': 'round-robin', 'timeout_seconds': 60, 'resume': False}},
                {'event': 'run_started', 'session_id': 'new', 'session_config':
                 {'policy': 'adaptive', 'timeout_seconds': 60, 'resume': True}},
            ]
            (path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
            result = report_run(path)
            self.assertEqual(result['session_timing_seconds']['result_validation_seconds'], .4)
            self.assertEqual(result['prior_sessions_without_configuration'], ['legacy'])
            self.assertEqual(result['prior_session_configuration_changes'], {'old': ['policy']})
            self.assertEqual(result['session_configuration']['policy'], 'adaptive')


if __name__ == '__main__':
    unittest.main()
