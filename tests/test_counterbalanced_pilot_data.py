import tempfile
import hashlib
from PIL import Image
import unittest
from collections import Counter
from pathlib import Path
from exposure.counterbalanced_pilot_data import VOCABULARY, generate

class CounterbalancedPilotTests(unittest.TestCase):
    def test_paired_geometry_balanced_roles_and_splits(self):
        with tempfile.TemporaryDirectory() as d:
            manifest=generate(Path(d)/'data',seed=1,train_pairs=24,test_pairs=12)
            rows=manifest['records']; pairs={}
            for r in rows: pairs.setdefault(r['pair_id'],{})[r['complexity']]=r
            for pair in pairs.values():
                self.assertEqual(pair['low']['objects'],pair['high']['objects'][:2])
                self.assertEqual(pair['low']['target'],pair['high']['target'])
                self.assertEqual(len(pair['high']['objects']),3)
                points=[(o['x'],o['y']) for o in pair['high']['objects']]
                for i,(x,y) in enumerate(points):
                    for a,b in points[i+1:]: self.assertTrue(abs(x-a)>12 or abs(y-b)>12)
            assignments = {}
            for r in rows:
                if r['split'] == 'train' and r['complexity'] == 'high':
                    target = (r['target']['color'], r['target']['shape'])
                    distractor = (r['objects'][1]['color'], r['objects'][1]['shape'])
                    assignments.setdefault(target, set()).add(distractor)
            self.assertTrue(any(len(v) > 1 for v in assignments.values()))
            for split in ('train','test'):
                high=[r for r in rows if r['split']==split and r['complexity']=='high']
                for role in range(3):
                    counts=Counter((r['objects'][role]['color'],r['objects'][role]['shape']) for r in high)
                    self.assertEqual(set(counts),set(VOCABULARY))
                    self.assertEqual(len(set(counts.values())),1)
            for row in rows:
                with Image.open(Path(d)/'data'/row['image']) as im:
                    self.assertEqual(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest(), row['pixel_sha256'])
            again=generate(Path(d)/'again',seed=1,train_pairs=24,test_pairs=12)
            self.assertEqual(rows,again['records'])
            train={r['pixel_sha256'] for r in rows if r['split']=='train'}
            test={r['pixel_sha256'] for r in rows if r['split']=='test'}
            self.assertFalse(train & test)

    def test_heldout_combinations_never_occur_in_training(self):
        with tempfile.TemporaryDirectory() as d:
            held={('red','circle'),('blue','square'),('green','triangle')}
            m=generate(Path(d)/'data',seed=2,train_pairs=18,test_pairs=12,heldout_combinations=held)
            train=[r for r in m['records'] if r['split']=='train']
            self.assertFalse(any((o['color'],o['shape']) in held for r in train for o in r['objects']))
            self.assertEqual({(r['target']['color'],r['target']['shape']) for r in m['records'] if r['split']=='test'},set(VOCABULARY))
            with self.assertRaises(FileExistsError): generate(Path(d)/'data',seed=2,train_pairs=18,test_pairs=12,heldout_combinations=held)

    def test_invalid_counts_and_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): generate(Path(d)/'a',seed=1,train_pairs=13,test_pairs=12)
            with self.assertRaises(ValueError): generate(Path(d)/'b',seed=1,train_pairs=12,test_pairs=12,heldout_combinations=[('red','missing')])

if __name__=='__main__': unittest.main()
