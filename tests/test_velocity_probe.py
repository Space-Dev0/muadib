import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is unavailable")
class VelocityProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import pilot_train, velocity_probe
        from exposure.identity_eps_finetune import _sample
        torch.set_num_threads(1)
        cls.torch, cls.pilot, cls.probe, cls.sample = torch, pilot_train, velocity_probe, staticmethod(_sample)

    def test_velocity_conversion_recovers_noise(self):
        torch = self.torch
        clean = torch.tensor([[[[-0.7, 0.3]]], [[[0.2, 0.9]]]])
        noise = torch.tensor([[[[0.8, -1.2]]], [[[0.4, 0.1]]]])
        alpha = torch.tensor([0.02, 0.8]).view(2, 1, 1, 1)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        v = self.probe.velocity_target(clean, noise, alpha)
        torch.testing.assert_close(self.probe.velocity_to_noise(noisy, v, alpha), noise)

    def test_velocity_oracle_through_real_sampler_interface(self):
        torch = self.torch
        beta, abar = self.pilot.schedule(8, torch.device("cpu"))
        clean = torch.full((1, 3, 64, 64), 0.25)

        class Oracle(torch.nn.Module):
            def forward(self, noisy, t, tokens, drop=None):
                a = abar[t, None, None, None]
                return (a.sqrt() * noisy - clean) / (1 - a).sqrt()

        wrapper = self.probe.VelocityAsNoise(Oracle(), abar)
        image = self.sample(wrapper, "standard", tokens=torch.zeros((1, 1, 4), dtype=torch.long),
                            beta=beta, alpha_bar=abar, seed=5, device=torch.device("cpu"))
        self.assertTrue(np.all(np.asarray(image) == 159))

    def test_matched_arms_and_velocity_checkpoint_guard(self):
        torch = self.torch
        definitions = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.png"
            Image.new("RGB", (64, 64), (255, 0, 0)).save(target)
            output = Path(directory) / "result"
            report = self.probe.run(definitions, target, output, rule="red", steps=2, batch_size=2,
                                    seed=4, init_seed=9, width=4, timesteps=8, lr=1e-4, device="cpu",
                                    deterministic=False, eval_times=[0, 7], eval_seeds=[0], sample_seeds=[1])
            self.assertEqual(report["initial_model_sha256"]["epsilon"], report["initial_model_sha256"]["velocity"])
            self.assertEqual(report["arms"]["epsilon"]["timestep_counts"], report["arms"]["velocity"]["timestep_counts"])
            ckpt = torch.load(output / "velocity.pt", map_location="cpu", weights_only=True)
            with self.assertRaises(ValueError):
                self.pilot.sample(ckpt, "red", seed=1, device="cpu", guidance=1.0)
            self.assertTrue((output / "velocity-1.png").is_file())


if __name__ == "__main__":
    unittest.main()
