import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class IndependentEvalTests(unittest.TestCase):
    def test_resume_preserves_partial_attempt_and_reuses_completed_output(self):
        from exposure import mosaic_independent_eval as module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / 'evaluation'; review = root / 'review'
            entries = {cell: {'checkpoint': f'{cell}.pt', 'sha256': cell,
                              'training_report_sha256': 'report'} for cell in module.CELLS}
            calls = []
            def fake_run(checkpoint, directory, *, seeds, guidance, device):
                directory.mkdir(parents=True); calls.append(checkpoint); rows = []
                for color in ('red', 'blue'):
                    for seed in seeds:
                        image = Image.new('RGB', (64, 64), (255, 255, 255))
                        buffer = io.BytesIO(); image.save(buffer, format='PNG'); raw = buffer.getvalue()
                        filename = f'{color}-{seed}.png'; (directory / filename).write_bytes(raw)
                        rows.append({'color': color, 'seed': seed, 'guidance': 1., 'file': filename,
                                     'sha256': hashlib.sha256(raw).hexdigest(),
                                     'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
                cell = Path(checkpoint).stem
                (directory / 'report.json').write_text(json.dumps({'checkpoint_sha256': cell,
                    'seeds': list(seeds), 'guidance': [1.], 'images': rows}))
            with mock.patch.object(module, 'checkpoints', return_value=entries), mock.patch.object(module, 'run', side_effect=fake_run):
                module.generate(root, output, review, device='cpu')
                self.assertEqual(len(calls), 4)
                module.generate(root, output, review, device='cpu')
                self.assertEqual(len(calls), 4)
                self.assertTrue((review / 'reviewer-4.zip').exists())
                self.assertTrue((output / 'COMPLETE').exists())


if __name__ == '__main__':
    unittest.main()
