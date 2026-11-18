import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from exposure.checker_calibration import generate, evaluate

class CheckerCalibrationTests(unittest.TestCase):
    def test_balanced_splits_determinism_and_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'one';q=Path(d)/'two'
            a=generate(p,seed=41,train_per_stratum=1,validation_per_stratum=1,test_per_stratum=1)
            b=generate(q,seed=41,train_per_stratum=1,validation_per_stratum=1,test_per_stratum=1)
            self.assertEqual(a,b)
            for split in ('train','validation','test'):
                truth=Counter(r['truth'] for r in a['records'] if r['split']==split)
                self.assertEqual(truth,{True:10,False:10})
            hashes=[r['clean_pixel_sha256'] for r in a['records']]
            self.assertEqual(len(hashes),len(set(hashes)))
            sha=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()
            result=evaluate(p,expected_sha256=sha,split='test')
            self.assertEqual(result['samples'],20)
            for groups in result['counts'].values():
                self.assertEqual(sum(groups['all'].values()),20)
                self.assertEqual(set(groups['all']),{f'{truth}_{decision}' for truth in ('positive','negative') for decision in ('yes','no','abstain')})
            with self.assertRaises(ValueError): evaluate(p,expected_sha256='0'*64)
            (p/a['records'][-1]['image']).write_bytes(b'bad')
            with self.assertRaises(ValueError):evaluate(p,expected_sha256=sha,split='test')

if __name__=='__main__':unittest.main()
