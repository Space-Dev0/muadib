import json
import tempfile
import unittest
from pathlib import Path

from ops import adaptive_inference
from ops import science_plan

SHA = 'a' * 64
OTHER = 'b' * 64


def make_job(i, stratum, checkpoint=SHA):
    return {'id': f'j{i}', 'stratum': stratum, 'checkpoint_sha256': checkpoint,
            'color': 'red' if i % 2 else 'blue', 'seed': i}


class SciencePlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'schema_version': 1, 'jobs': [make_job(i, 'A' if i <= 6 else 'B')
                                                     for i in range(1, 13)]}
        self.workers = [{'id': f'w{i}', 'port': 10000 + i, 'checkpoint_sha256': SHA}
                        for i in range(1, 4)]

    def test_balances_each_stratum_without_changing_scientific_jobs(self):
        design = science_plan.create_design(self.plan, self.workers, audit_per_stratum=2)
        assignments = design['manifest']['assignments']
        self.assertEqual({x['job_id'] for x in assignments}, {j['id'] for j in self.plan['jobs']})
        self.assertEqual(len(assignments), len(self.plan['jobs']))
        for stratum in ('A', 'B'):
            counts = [sum(x['stratum'] == stratum and x['worker_id'] == w['id'] for x in assignments)
                      for w in self.workers]
            self.assertEqual(counts, [2, 2, 2])
        emitted = [j for name, plan in design['plans'].items() if name.startswith('base/')
                   for j in plan['jobs']]
        self.assertEqual(sorted(emitted, key=lambda j: j['id']),
                         sorted(self.plan['jobs'], key=lambda j: j['id']))
        self.assertEqual(set(design['base_assignments']['assignments']), {j['id'] for j in emitted})

    def test_replay_preserves_exact_job_and_is_separate_from_base(self):
        design = science_plan.create_design(self.plan, self.workers, audit_per_stratum=2, selection_seed=12)
        jobs = {j['id']: j for j in self.plan['jobs']}
        self.assertEqual(len(design['manifest']['audit_pairs']), 4)
        for pair in design['manifest']['audit_pairs']:
            original = jobs[pair['job_id']]
            self.assertEqual(pair['primary_worker'], pair['repeat_worker'])
            for phase, worker in [('base', pair['primary_worker']), ('repeat', pair['repeat_worker']),
                                  *[('cross', w) for w in pair['comparison_workers']]]:
                self.assertIn(original, design['plans'][f'{phase}/{worker}-plan.json']['jobs'])
            self.assertEqual(len(pair['comparison_workers']), 2)
        for plan in design['plans'].values():
            adaptive_inference.validate_plan(plan)

    def test_selection_is_deterministic_and_input_order_independent(self):
        a = science_plan.create_design(self.plan, self.workers, audit_per_stratum=2, selection_seed=9)
        shuffled = {'schema_version': 1, 'jobs': list(reversed(self.plan['jobs']))}
        b = science_plan.create_design(shuffled, list(reversed(self.workers)), audit_per_stratum=2,
                                       selection_seed=9)
        self.assertEqual(a['manifest']['assignments'], b['manifest']['assignments'])
        self.assertEqual(a['manifest']['audit_pairs'], b['manifest']['audit_pairs'])

    def test_checkpoint_compatibility_and_single_worker_audit(self):
        plan = {'schema_version': 1, 'jobs': [make_job(1, 'A', OTHER), make_job(2, 'A', SHA)]}
        workers = self.workers[:1] + [{'id': 'other', 'port': 12345, 'checkpoint_sha256': OTHER}]
        design = science_plan.create_design(plan, workers, audit_per_stratum=2)
        self.assertEqual({a['job_id']: a['worker_id'] for a in design['manifest']['assignments']},
                         {'j1': 'other', 'j2': 'w1'})
        self.assertTrue(all(not p['comparison_workers'] for p in design['manifest']['audit_pairs']))
        with self.assertRaisesRegex(ValueError, 'compatible'):
            science_plan.create_design(plan, self.workers)

    def test_write_exclusive_and_reject_invalid_parameters(self):
        design = science_plan.create_design(self.plan, self.workers)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'design'
            science_plan.write_design(out, design)
            self.assertEqual(json.loads((out / 'manifest.json').read_text()), design['manifest'])
            self.assertEqual(json.loads((out / 'base-plan.json').read_text()), self.plan)
            self.assertEqual(json.loads((out / 'base-assignments.json').read_text()),
                             design['base_assignments'])
            with self.assertRaises(FileExistsError):
                science_plan.write_design(out, design)
        for bad in (-1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                science_plan.create_design(self.plan, self.workers, audit_per_stratum=bad)
        case_collision = [self.workers[0], dict(self.workers[1], id='W1')]
        with self.assertRaisesRegex(ValueError, 'case-insensitive'):
            science_plan.create_design(self.plan, case_collision)


if __name__ == '__main__':
    unittest.main()
