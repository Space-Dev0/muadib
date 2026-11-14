import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, 'PyTorch is optional')
class ResumeTest(unittest.TestCase):
    def test_cpu_resume_matches_uninterrupted(self):
        from exposure.mosaic_resumable_probe import train
        dataset=Path('outputs/mosaic-data-gpu-feasibility-81300')
        parent=Path('outputs/mosaic-warmstart-probe-81350/warmstart-plus-30000.pt')
        if not dataset.exists() or not parent.exists(): self.skipTest('GPU experiment artifacts are absent')
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a'; b=Path(tmp)/'b'
            train(dataset,parent,a,steps=2,interval=1,seed=91,device='cpu')
            train(dataset,parent,b,steps=1,interval=1,seed=91,device='cpu')
            train(dataset,parent,b,steps=2,interval=1,seed=91,device='cpu',resume=b/'resumable-plus-1.pt')
            x=torch.load(a/'resumable-plus-2.pt',weights_only=True,map_location='cpu')['model']
            y=torch.load(b/'resumable-plus-2.pt',weights_only=True,map_location='cpu')['model']
            self.assertTrue(all(torch.equal(x[k], y[k]) for k in x))
