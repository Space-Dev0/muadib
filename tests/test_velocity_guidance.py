import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is unavailable")
class VelocityGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import velocity_guidance, velocity_online
        torch.set_num_threads(1)
        cls.torch, cls.guidance, cls.online = torch, velocity_guidance, velocity_online
        cls.rules = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]}
        cls.checkpoint, _ = velocity_online.train(
            cls.rules, steps=1, batch_size=1, seed=3, device="cpu", width=4,
            timesteps=8, lr=1e-4, dropout=0.1, ema_decay=0.9,
            max_attempts=10000, deterministic=True)

    def test_guidance_formula_and_conversion(self):
        torch = self.torch

        class Stub(torch.nn.Module):
            def forward(self, noisy, timestep, tokens, drop=None):
                return torch.full_like(noisy, 2.0 if drop is None else -1.0)

        noisy = torch.tensor([[[[0.25]]]])
        alpha = torch.tensor([0.36])
        timestep = torch.tensor([0])
        tokens = torch.zeros((1, 1, 4), dtype=torch.long)
        for scale in (0.0, 1.0, 2.5):
            result = self.guidance.GuidedVelocity(Stub(), alpha, scale)(noisy, timestep, tokens)
            expected_velocity = -1.0 + scale * 3.0
            self.assertAlmostEqual(result.item(), 0.6 * expected_velocity + 0.8 * 0.25, places=6)

    def test_scale_one_matches_existing_sampler_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "first"
            result = self.guidance.sweep(self.checkpoint, output, rules=["red"],
                                         seeds=[4], scales=[1.0, 2.0], device="cpu")
            from PIL import Image
            expected = self.online.sample(self.checkpoint, "red", seed=4, device="cpu")
            self.assertEqual(Image.open(result["samples"][0]["path"]).tobytes(), expected.tobytes())
            again = self.guidance.sweep(self.checkpoint, Path(root) / "second", rules=["red"],
                                        seeds=[4], scales=[1.0, 2.0], device="cpu")
            for first, second in zip(result["samples"], again["samples"], strict=True):
                self.assertEqual(Path(first["path"]).read_bytes(), Path(second["path"]).read_bytes())
            self.assertEqual(json.loads((output / "report.json").read_text())["scales"], [1.0, 2.0])

    def test_invalid_inputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "images"
            for scales in ([float("nan")], [-1.0], [1.0, 1.0], [True]):
                with self.assertRaises(ValueError):
                    self.guidance.sweep(self.checkpoint, output, rules=["red"], seeds=[0], scales=scales, device="cpu")
            with self.assertRaises(ValueError):
                self.guidance.sweep(self.checkpoint, output, rules=["unknown"], seeds=[0], scales=[1.0], device="cpu")
            bad = copy.deepcopy(self.checkpoint)
            bad["diagnostic"]["requires_custom_sampler"] = False
            with self.assertRaises(ValueError):
                self.guidance.sweep(bad, output, rules=["red"], seeds=[0], scales=[1.0], device="cpu")
            checkpoint_path = Path(root) / "model.pt"
            self.torch.save(self.checkpoint, checkpoint_path)
            before = checkpoint_path.read_bytes()
            result = subprocess.run([sys.executable, "-m", "exposure.velocity_guidance",
                                     "--checkpoint", str(checkpoint_path), "--output", str(checkpoint_path),
                                     "--rules", "red", "--seeds", "0", "--scales", "1"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(before, checkpoint_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
