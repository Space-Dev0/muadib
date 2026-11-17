import unittest
from exposure.templates_v2 import build
from exposure.benchmark_audit import audit
from exposure.render import COLORS, SHAPES

class FactorialTemplateTests(unittest.TestCase):
    def test_counterbalanced_with_partition_and_witness_binding(self):
        rules, manifest = build()
        self.assertEqual(len(rules), 288)
        self.assertEqual(len(manifest['train']), 216)
        self.assertEqual(len(manifest['heldout']), 72)
        self.assertFalse(set(manifest['train']) & set(manifest['heldout']))
        self.assertEqual(set(rules), set(manifest['train']) | set(manifest['heldout']))
        for partition in ('train', 'heldout'):
            counts = manifest['coverage'][partition]
            self.assertEqual(counts['left_of'], counts['above'])
            self.assertTrue(all(counts[f'pair:{c}/{s}'] > 0 for c in COLORS for s in SHAPES))
        train = set(manifest['train'])
        for name in rules:
            if '_left_of_' in name:
                mate = name.replace('_left_of_', '_above_')
                self.assertIn(mate, rules)
                self.assertEqual(name in train, mate in train)
        for predicates in rules.values():
            self.assertEqual({p['variable'] for p in predicates[:-1]}, {'a', 'b'})
            self.assertEqual({p['field'] for p in predicates[:-1] if p['variable']=='a'}, {'color','shape'})
            self.assertEqual({p['field'] for p in predicates[:-1] if p['variable']=='b'}, {'color','shape'})
        result = audit(rules, split_manifest=manifest)
        self.assertEqual(result['pair_coverage']['same_object_pair_count'], 12)
        self.assertFalse(result['endpoint_binding']['confounded_cells'])
        self.assertFalse(result['confirmatory_ready'])
        self.assertTrue(result['held_out_split']['present'])
        self.assertNotIn('no_held_out_template_split', [v['id'] for v in result['blocking_findings']])
        with self.assertRaises(ValueError):
            audit(rules, split_manifest={'train':manifest['train'], 'heldout':manifest['train']})
        renamed = dict(rules)
        renamed['duplicate'] = list(renamed[manifest['train'][0]])
        bad = {'train':manifest['train'], 'heldout':manifest['heldout'] + ['duplicate']}
        with self.assertRaisesRegex(ValueError, 'semantically duplicate'):
            audit(renamed, split_manifest=bad)
        self.assertIn('saturated_relation_singletons', [v['id'] for v in result['blocking_findings']])

if __name__ == '__main__': unittest.main()
