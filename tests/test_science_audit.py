import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from ops import science_audit, science_plan


class ScienceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        digest = 'a' * 64
        self.plan = {'schema_version': 1, 'jobs': [
            {'id': f'{color}-{seed}', 'stratum': 'balanced', 'checkpoint_sha256': digest,
             'color': color, 'seed': seed}
            for color in ('red', 'blue') for seed in (1, 2)]}
        workers = [{'id': 'gpu-a', 'port': 50101, 'checkpoint_sha256': digest},
                   {'id': 'gpu-b', 'port': 50102, 'checkpoint_sha256': digest}]
        design = science_plan.create_design(self.plan, workers, audit_per_stratum=1, selection_seed=9)
        self.manifest = design['manifest']
        self.design = self.root / 'design'
        science_plan.write_design(self.design, design)
        self.runs = self.root / 'runs'
        self.pair = self.manifest['audit_pairs'][0]
        self.replay_worker = self.pair['comparison_workers'][0]
        self.job_id = self.pair['job_id']
        self.write_run_plans(design)

    def write_run_plans(self, design):
        for name, plan in design['plans'].items():
            phase, filename = name.split('/')
            worker = filename.removesuffix('-plan.json')
            run = self.runs / phase / worker
            run.mkdir(parents=True)
            (run / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')

    def create_result(self, phase, worker, job, *, pixel=(2, 5, 7), actual_worker=None):
        run = self.runs / phase / worker
        target = run / 'results' / job['id']
        target.mkdir(parents=True)
        buffer = io.BytesIO()
        Image.new('RGB', (64, 64), pixel).save(buffer, format='PNG')
        data = buffer.getvalue()
        (target / 'image.png').write_bytes(data)
        (target / 'record.json').write_text(json.dumps({
            'job': job, 'worker_id': actual_worker or worker, 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}), encoding='utf-8')

    def fill_results(self):
        for rel in self.manifest['plan_files']:
            phase, filename = rel.split('/')
            worker = filename.removesuffix('-plan.json')
            plan = json.loads((self.design / rel).read_text())
            for job in plan['jobs']:
                pixel = (9, 5, 7) if phase == 'cross' else (2, 5, 7)
                self.create_result(phase, worker, job, pixel=pixel)

    def run_audit(self):
        return science_audit.audit(self.design / 'manifest.json', self.runs)

    def test_primary_counts_exclude_replays_and_pair_differences(self):
        self.fill_results()
        report = self.run_audit()
        base = report['strata']['balanced']
        self.assertEqual((base['planned_primary'], base['completed_primary'], base['missing_primary']), (4, 4, 0))
        self.assertEqual(base['completed_by_worker'], {'gpu-a': 2, 'gpu-b': 2})
        pairs = {p['kind']: p for p in report['comparisons']}
        self.assertEqual(pairs['same_worker_repeat']['identical_png_bytes'], 1)
        self.assertEqual(pairs['cross_worker_replay']['different_png_bytes'], 1)
        self.assertTrue(any('not a test of visual' in note for note in report['notes']))

    def test_missing_and_wrong_worker_are_not_compared(self):
        self.fill_results()
        primary = self.pair['primary_worker']
        original = self.runs / 'base' / primary / 'results' / self.job_id
        (original / 'image.png').unlink()
        (original / 'record.json').unlink()
        original.rmdir()
        report = self.run_audit()
        self.assertEqual(report['strata']['balanced']['missing_primary'], 1)
        self.assertTrue(all(p['missing_primary'] == 1 and p['complete_pairs'] == 0
                            for p in report['comparisons']))
        # Restore primary, but mark the cross-worker result as a wrong worker.
        job = next(j for j in self.plan['jobs'] if j['id'] == self.job_id)
        self.create_result('base', primary, job)
        record = self.runs / 'cross' / self.replay_worker / 'results' / self.job_id / 'record.json'
        data = json.loads(record.read_text()); data['worker_id'] = primary
        record.write_text(json.dumps(data))
        cross = next(p for p in self.run_audit()['comparisons'] if p['kind'] == 'cross_worker_replay')
        self.assertEqual(cross['worker_assignment_mismatches'], 1)
        self.assertEqual(cross['complete_pairs'], 0)

    def test_tampered_png_and_wrong_replay_plan_rejected(self):
        self.fill_results()
        path = self.runs / 'cross' / self.replay_worker / 'results' / self.job_id / 'image.png'
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'validation failed'):
            self.run_audit()
        # Even an empty run cannot silently use a different plan.
        path.unlink()
        run_plan = self.runs / 'cross' / self.replay_worker / 'plan.json'
        data = json.loads(run_plan.read_text()); data['jobs'][0]['seed'] += 10
        run_plan.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'run plan differs'):
            self.run_audit()

    def test_manifest_requires_all_original_jobs_and_unmodified_base_plan(self):
        manifest_path = self.design / 'manifest.json'
        data = json.loads(manifest_path.read_text()); data['assignments'].pop()
        manifest_path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'do not match the frozen base plan'):
            self.run_audit()

    def test_unstarted_runs_count_as_missing(self):
        report = self.run_audit()
        self.assertEqual(report['strata']['balanced']['missing_primary'], 4)
        self.assertTrue(all(p['incomplete_pairs'] == 1 for p in report['comparisons']))


if __name__ == '__main__':
    unittest.main()
