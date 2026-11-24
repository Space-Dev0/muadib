import io
import hashlib
import struct
import zlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from ops import adaptive_inference as ai


SHA = 'a' * 64


def png():
    buffer = io.BytesIO()
    Image.new('RGB', (64, 64), (1, 2, 3)).save(buffer, format='PNG')
    return buffer.getvalue()


def plan():
    return {'schema_version': 1, 'jobs': [
        {'id': 'one', 'stratum': 'red', 'checkpoint_sha256': SHA, 'color': 'red', 'seed': 1},
        {'id': 'two', 'stratum': 'blue', 'checkpoint_sha256': SHA, 'color': 'blue', 'seed': 2},
    ]}


def worker(name, port):
    return {'id': name, 'port': port, 'checkpoint_sha256': SHA}


class TestAdaptiveInference(unittest.TestCase):
    def test_session_configuration_and_setup_are_recorded_without_probe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            first_worker = dict(worker('a', 10001), probe_file='/secret/probe.json', probe_port=11001,
                                estimate_seconds=12)
            def partial(endpoint, job, timeout):
                if job['seed'] == 2:
                    raise ConnectionRefusedError('offline')
                return png(), .5, {}
            with patch.object(ai.distributed_inference, 'request', side_effect=partial):
                ai.run(plan(), [first_worker], output, policy='round-robin', timeout=42,
                       telemetry_max_age=7)
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), .5, {})):
                summary = ai.run(plan(), [worker('b', 10002)], output, policy='adaptive',
                                 timeout=50, resume=True)
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            sessions = [e['session_config'] for e in events if e['event'] == 'run_started']
            self.assertEqual([s['policy'] for s in sessions], ['round-robin', 'adaptive'])
            self.assertEqual(sessions[0]['timeout_seconds'], 42)
            self.assertEqual(sessions[0]['telemetry_max_age_seconds'], 7)
            self.assertEqual(sessions[0]['workers'][0]['probe_port'], 11001)
            self.assertTrue(sessions[0]['workers'][0]['probe_enabled'])
            self.assertNotIn('/secret/probe.json', json.dumps(sessions))
            self.assertEqual(sessions[1]['workers'][0]['id'], 'b')
            self.assertEqual(len(sessions[0]['coordinator_source_sha256']), 64)
            self.assertGreaterEqual(summary['setup_seconds'], summary['result_validation_seconds'])
            self.assertGreaterEqual(summary['session_elapsed_before_summary_seconds'], summary['segment_wall_seconds'])

    def test_bad_probe_paths_are_ignored_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fifo = root / 'fifo'
            large = root / 'large'
            with large.open('wb') as f:
                f.truncate(ai._MAX_PROBE_BYTES + 1)
            paths = [large]
            if hasattr(os, 'mkfifo'):
                os.mkfifo(fifo)
                paths.append(fifo)
                link = root / 'link'
                link.symlink_to(fifo)
                paths.append(link)
            for path in paths:
                with self.subTest(path=path):
                    w = dict(worker('a', 10001), probe_file=str(path), probe_port=10002)
                    self.assertIsNone(ai._fresh_rtt(w, 15))

    def test_local_publish_failure_is_not_an_uncertain_remote_failure(self):
        single = plan()
        single['jobs'] = single['jobs'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), .5, {})):
                with patch.object(ai, '_publish', side_effect=OSError('disk full')):
                    result = ai.run(single, [worker('a', 10001)], output)
                self.assertEqual(result['missing'], 1)
                failed = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()
                          if json.loads(line).get('event') == 'attempt_failed'][-1]
                self.assertEqual(failed['failure_phase'], 'publication')
                self.assertFalse(failed['uncertain'])
                result = ai.run(single, [worker('a', 10001)], output, resume=True)
                self.assertEqual(result['completed'], 1)

    def test_torn_final_journal_is_archived_and_requires_retry_acknowledgment(self):
        single = plan()
        single['jobs'] = single['jobs'][:1]
        workers = [worker('a', 10001)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', side_effect=ConnectionRefusedError('offline')):
                ai.run(single, workers, output)
            events = output / 'events.jsonl'
            events.write_bytes(events.read_bytes() + b'{"event":"attempt_sta')
            original = events.read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'still run remotely'):
                ai.run(single, workers, output, resume=True)
            archives = list(output.glob('events.torn-*.jsonl'))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), original)
            self.assertTrue(any(json.loads(line)['event'] == 'journal_recovered'
                                for line in events.read_text().splitlines()))
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), .5, {})):
                result = ai.run(single, workers, output, resume=True, allow_uncertain_retry=True)
            self.assertEqual(result['completed'], 1)

    def test_malformed_complete_journal_line_is_not_discarded(self):
        single = plan()
        single['jobs'] = single['jobs'][:1]
        workers = [worker('a', 10001)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', side_effect=ConnectionRefusedError('offline')):
                ai.run(single, workers, output)
            events = output / 'events.jsonl'
            original = events.read_bytes() + b'{bad}\n{"event":"run_started"}\n'
            events.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'malformed complete or middle'):
                ai.run(single, workers, output, resume=True)
            self.assertEqual(events.read_bytes(), original)
            self.assertEqual(list(output.glob('events.torn-*.jsonl')), [])

    def test_valid_final_event_without_newline_is_preserved(self):
        single = plan()
        single['jobs'] = single['jobs'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), .5, {})):
                ai.run(single, [worker('a', 10001)], output)
            events = output / 'events.jsonl'
            original = events.read_bytes().removesuffix(b'\n')
            events.write_bytes(original)
            result = ai.run(single, [worker('a', 10001)], output, resume=True)
            self.assertEqual(result['completed'], 1)
            self.assertEqual(next(output.glob('events.torn-*.jsonl')).read_bytes(), original)
            self.assertTrue(any(json.loads(line)['event'] == 'journal_recovered'
                                for line in events.read_text().splitlines()))

    def test_failover_preserves_fixed_plan_and_strata(self):
        workers = [worker('first', 10001), worker('second', 10002)]
        def request(endpoint, job, timeout):
            if endpoint[1] == 10001:
                raise ConnectionRefusedError('offline')
            return png(), 0.25, {'device': 'cpu'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(ai.distributed_inference, 'request', side_effect=request):
            output = Path(tmp) / 'run'
            result = ai.run(plan(), workers, output)
            self.assertEqual((result['planned'], result['completed'], result['missing']), (2, 2, 0))
            self.assertEqual(result['strata']['red']['completed'], 1)
            self.assertEqual(result['strata']['blue']['completed'], 1)
            self.assertEqual(json.loads((output / 'plan.json').read_text()), plan())
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(event['event'] == 'attempt_failed' for event in events))
            self.assertEqual(sum(event['event'] == 'attempt_succeeded' for event in events), 2)
            for job in plan()['jobs']:
                self.assertIsNotNone(ai._load_valid_result(output, job))

    def test_resume_skips_completed_and_rejects_tampering(self):
        workers = [worker('first', 10001)]
        def partial(endpoint, job, timeout):
            if job['seed'] == 2:
                raise ConnectionRefusedError('offline')
            return png(), 0.5, {}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', side_effect=partial):
                first = ai.run(plan(), workers, output)
            self.assertEqual((first['completed'], first['missing']), (1, 1))
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})) as call:
                second = ai.run(plan(), workers, output, resume=True)
            self.assertEqual((second['completed'], second['missing'], second['reused_jobs']), (2, 0, 1))
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[1]['seed'], 2)
            altered = plan()
            altered['jobs'][0]['seed'] = 99
            with self.assertRaisesRegex(ValueError, 'differs'):
                ai.run(altered, workers, output, resume=True)
            (output / 'results' / 'one' / 'image.png').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'validation failed'):
                ai.run(plan(), workers, output, resume=True)

    def test_fresh_rtt_only_and_adaptive_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workers = [worker('slow', 10001), worker('fast', 10002)]
            for w, rtt in zip(workers, [80, 5]):
                w.update(probe_file=str(root / (w['id'] + '.json')), probe_port=w['port'] + 100)
                Path(w['probe_file']).write_text(json.dumps({
                    'schema_version': 1,
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'endpoint': {'host': '127.0.0.1', 'port': w['probe_port']},
                    'configuration': {'count': 5, 'interval_seconds': .1, 'timeout_seconds': 2,
                                      'payload_bytes_each_direction': 1024},
                    'summary': {'attempts': 5, 'successes': 5, 'failed_attempts': 0, 'echo_rtt_median_ms': rtt},
                    'attempts': [{'sequence': i, 'status': 'ok', 'echo_rtt_ms': rtt} for i in range(5)],
                }))
            single = plan()
            single['jobs'] = single['jobs'][:1]
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})) as call:
                ai.run(single, workers, root / 'run', policy='adaptive')
            self.assertEqual(call.call_args.args[0][1], 10002)
            data = json.loads(Path(workers[1]['probe_file']).read_text())
            data['created_utc'] = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            Path(workers[1]['probe_file']).write_text(json.dumps(data))
            self.assertIsNone(ai._fresh_rtt(workers[1], 15))
            data['created_utc'] = datetime.now(timezone.utc).isoformat()
            data['summary']['attempts'] = 6
            data['summary']['failed_attempts'] = 1
            Path(workers[1]['probe_file']).write_text(json.dumps(data))
            self.assertIsNone(ai._fresh_rtt(workers[1], 15))
            data['summary']['attempts'] = 5
            data['summary']['failed_attempts'] = 0
            data['endpoint']['port'] += 1
            Path(workers[1]['probe_file']).write_text(json.dumps(data))
            self.assertIsNone(ai._fresh_rtt(workers[1], 15))

    def test_probe_requires_schema_five_consistent_successes_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.json'
            w = dict(worker('w', 10001), probe_file=str(path), probe_port=10002)
            valid = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                     'endpoint': {'host': '127.0.0.1', 'port': 10002},
                     'configuration': {'count': 5, 'interval_seconds': .1, 'timeout_seconds': 2,
                                       'payload_bytes_each_direction': 1024},
                     'summary': {'attempts': 5, 'successes': 5, 'failed_attempts': 0,
                                 'echo_rtt_median_ms': 4.2},
                     'attempts': [{'sequence': i, 'status': 'ok', 'echo_rtt_ms': 4.2} for i in range(5)]}
            path.write_text(json.dumps(valid))
            self.assertEqual(ai._fresh_rtt(w, 15), 4.2)
            for mutation in ('schema', 'count', 'timeout', 'payload', 'attempt', 'median'):
                value = json.loads(json.dumps(valid))
                if mutation == 'schema': value['schema_version'] = True
                if mutation == 'count': value['configuration']['count'] = 1
                if mutation == 'timeout': value['configuration']['timeout_seconds'] = 0
                if mutation == 'payload': value['configuration']['payload_bytes_each_direction'] = 0
                if mutation == 'attempt': value['attempts'][2]['status'] = 'error'
                if mutation == 'median': value['summary']['echo_rtt_median_ms'] = 5
                path.write_text(json.dumps(value))
                with self.subTest(mutation=mutation):
                    self.assertIsNone(ai._fresh_rtt(w, 15))

    def test_decompression_bomb_is_uncertain_and_resume_error_is_clean(self):
        workers = [worker('a', 10001)]
        single = plan()
        single['jobs'] = single['jobs'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request',
                              side_effect=Image.DecompressionBombError('oversized')):
                result = ai.run(single, workers, output)
            self.assertEqual(result['missing'], 1)
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            failed = next(e for e in events if e['event'] == 'attempt_failed')
            self.assertTrue(failed['uncertain'])
            self.assertEqual(failed['error_type'], 'DecompressionBombError')
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})):
                ai.run(single, workers, output, resume=True, allow_uncertain_retry=True)
            with patch.object(ai.Image, 'open', side_effect=Image.DecompressionBombError('oversized')):
                with self.assertRaisesRegex(ValueError, 'DecompressionBombError'):
                    ai.run(single, workers, output, resume=True)

    def test_corrupt_png_syntax_error_is_uncertain_and_resume_error_is_clean(self):
        workers = [worker('a', 10001)]
        single = plan()
        single['jobs'] = single['jobs'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', side_effect=SyntaxError('bad CRC')):
                result = ai.run(single, workers, output)
            self.assertEqual(result['missing'], 1)
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            failed = next(e for e in events if e['event'] == 'attempt_failed')
            self.assertTrue(failed['uncertain'])
            self.assertEqual(failed['error_type'], 'SyntaxError')
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})):
                ai.run(single, workers, output, resume=True, allow_uncertain_retry=True)
            with patch.object(ai.Image, 'open', side_effect=SyntaxError('bad CRC')):
                with self.assertRaisesRegex(ValueError, 'SyntaxError'):
                    ai.run(single, workers, output, resume=True)

    def test_png_with_valid_crc_but_invalid_pixels_is_rejected(self):
        def chunk(kind, data):
            return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
        invalid = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 64, 64, 8, 2, 0, 0, 0))
                   + chunk(b'IDAT', zlib.compress(b'')) + chunk(b'IEND', b''))
        with Image.open(io.BytesIO(invalid)) as image:
            image.verify()  # CRC/framing valid; decoder must still reject this image.
        workers = [worker('a', 10001)]
        single = plan()
        single['jobs'] = single['jobs'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', return_value=(invalid, 0.5, {})):
                result = ai.run(single, workers, output)
            self.assertEqual(result['missing'], 1)
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})):
                ai.run(single, workers, output, resume=True)
            directory = output / 'results' / 'one'
            (directory / 'image.png').write_bytes(invalid)
            record = json.loads((directory / 'record.json').read_text())
            record.update(sha256=hashlib.sha256(invalid).hexdigest(), bytes=len(invalid))
            (directory / 'record.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'invalid PNG'):
                ai.run(single, workers, output, resume=True)

    def test_resume_rejects_symlinked_metadata(self):
        single = plan()
        single['jobs'] = single['jobs'][:1]
        workers = [worker('a', 10001)]
        for name in ('plan.json', 'events.jsonl', 'summary.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'run'
                with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})):
                    ai.run(single, workers, output)
                path = output / name
                external = Path(tmp) / 'external'
                external.write_bytes(path.read_bytes())
                expected = external.read_bytes()
                path.unlink()
                path.symlink_to(external)
                with self.assertRaisesRegex(ValueError, 'symlink is not allowed'):
                    ai.run(single, workers, output, resume=True)
                self.assertEqual(external.read_bytes(), expected)

    def test_resume_rejects_oversized_result_before_read(self):
        single = plan()
        single['jobs'] = single['jobs'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})):
                ai.run(single, [worker('a', 10001)], output)
            image = output / 'results' / 'one' / 'image.png'
            with image.open('wb') as file:
                file.truncate(ai.distributed_inference.MAX_IMAGE + 1)
            with self.assertRaisesRegex(ValueError, 'invalid image size'):
                ai.run(single, [worker('a', 10001)], output, resume=True)

    def test_resume_rejects_invalid_or_oversized_record_without_blocking(self):
        single = plan()
        single['jobs'] = single['jobs'][:1]
        variants = ['oversized', 'directory', 'invalid_json']
        if hasattr(os, 'mkfifo'):
            variants.append('fifo')
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'run'
                with patch.object(ai.distributed_inference, 'request', return_value=(png(), .5, {})):
                    ai.run(single, [worker('a', 10001)], output)
                record = output / 'results' / 'one' / 'record.json'
                record.unlink()
                if variant == 'oversized':
                    with record.open('wb') as f:
                        f.truncate(ai._MAX_RESULT_RECORD_BYTES + 1)
                elif variant == 'directory':
                    record.mkdir()
                elif variant == 'fifo':
                    os.mkfifo(record)
                else:
                    record.write_text('{bad')
                with self.assertRaisesRegex(ValueError, 'invalid record'):
                    ai.run(single, [worker('a', 10001)], output, resume=True)

    def test_cleanup_does_not_remove_replacement_lock_or_mask_error(self):
        single = plan()
        single['jobs'] = single['jobs'][:1]
        for replace in (False, True):
            with self.subTest(replace=replace), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'run'
                spare = Path(tmp) / 'replacement'
                spare.write_text('other process')
                def fail(events, event):
                    lock = output / '.coordinator.lock'
                    if replace:
                        os.replace(spare, lock)
                    else:
                        lock.unlink()
                    raise RuntimeError('original error')
                with patch.object(ai, '_event', side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, 'original error'):
                        ai.run(single, [worker('a', 10001)], output)
                if replace:
                    self.assertEqual((output / '.coordinator.lock').read_text(), 'other process')

    def test_non_object_result_record_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})):
                ai.run(plan(), [worker('a', 10001)], output)
            (output / 'results' / 'one' / 'record.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'result validation failed'):
                ai.run(plan(), [worker('a', 10001)], output, resume=True)

    def test_abandoned_attempt_keeps_original_session_and_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            single = plan()
            single['jobs'] = single['jobs'][:1]
            with patch.object(ai.distributed_inference, 'request', side_effect=ConnectionRefusedError('offline')):
                ai.run(single, [worker('a', 10001)], output)
            with (output / 'events.jsonl').open('a') as events:
                events.write(json.dumps({'event': 'attempt_started', 'attempt_id': 'dangling',
                                         'session_id': 'prior', 'worker_id': 'a',
                                         'stratum': 'red', 'job_id': 'one'}) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'still run remotely'):
                ai.run(single, [worker('a', 10001)], output, resume=True)
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            abandoned = next(e for e in events if e.get('event') == 'attempt_abandoned')
            self.assertEqual((abandoned['session_id'], abandoned['worker_id'], abandoned['stratum']),
                             ('prior', 'a', 'red'))

    def test_timeout_quarantines_worker_and_requires_explicit_retry(self):
        workers = [worker('first', 10001), worker('second', 10002)]
        single = plan()
        single['jobs'] = single['jobs'][:1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            with patch.object(ai.distributed_inference, 'request', side_effect=TimeoutError('timeout')) as request:
                result = ai.run(single, workers, output)
            self.assertEqual(result['missing'], 1)
            self.assertEqual(request.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, 'still run remotely'):
                ai.run(single, workers, output, resume=True)
            with patch.object(ai.distributed_inference, 'request', return_value=(png(), 0.5, {})):
                result = ai.run(single, workers, output, resume=True, allow_uncertain_retry=True)
            self.assertEqual(result['completed'], 1)

    def test_validation_rejects_duplicate_or_incompatible_work(self):
        for version in (True, 1.0):
            p = plan()
            p['schema_version'] = version
            with self.assertRaises(ValueError):
                ai.validate_plan(p)
        p = plan()
        p['jobs'][1] = dict(p['jobs'][0], id='other', stratum='other')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            ai.validate_plan(p)
        p = plan()
        p['jobs'][0]['id'], p['jobs'][1]['id'] = 'Case', 'case'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            ai.validate_plan(p)
        p = plan()
        p['jobs'][0]['id'] = '../escape'
        with self.assertRaises(ValueError):
            ai.validate_plan(p)
        with tempfile.TemporaryDirectory() as tmp:
            w = worker('other', 10001)
            w['checkpoint_sha256'] = 'b' * 64
            with self.assertRaisesRegex(ValueError, 'compatible'):
                ai.run(plan(), [w], Path(tmp) / 'run')


if __name__ == '__main__':
    unittest.main()
