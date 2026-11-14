import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional')
class StepProbeTests(unittest.TestCase):
    def test_same_seed_and_weights_for_all_step_counts(self):
        from PIL import Image
        from exposure.mosaic_multiscale_probe import MultiscaleVelocity
        from exposure.mosaic_step_probe import run
        model = MultiscaleVelocity(4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cp = root/'checkpoint.pt'
            torch.save({'kind':'mosaic_multiscale_probe','step':20000,'width':4,
                        'state_dict':model.state_dict()}, cp)
            calls=[]
            def fake_sample(model, label, seed, steps, device):
                calls.append((label,seed,steps,device))
                return Image.new('RGB',(64,64))
            with patch('exposure.mosaic_step_probe.sample', side_effect=fake_sample):
                result=run(cp,root/'output',seed=123)
            self.assertEqual(calls, [(label,123,steps,'cpu') for label in ('red','blue') for steps in (50,200,500)])
            self.assertEqual(len(result['results']),6)
            self.assertEqual(json.loads((root/'output'/'report.json').read_text())['checkpoint_step'],20000)
            with self.assertRaises(FileExistsError):
                run(cp,root/'output',seed=123)
