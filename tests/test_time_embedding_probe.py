import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
class TimeEmbeddingProbeTests(unittest.TestCase):
    def test_initial_non_time_parameters_match_and_sinusoidal_is_deterministic(self):
        import torch
        from exposure.time_embedding_probe import build_models

        learned, continuous = build_models(4, 8, 19, torch.device("cpu"))
        for name, value in learned.state_dict().items():
            if not name.startswith("time."):
                torch.testing.assert_close(value, continuous.state_dict()[name], rtol=0, atol=0)
        t = torch.tensor([0, 2, 7])
        torch.testing.assert_close(continuous.time(t), continuous.time(t), rtol=0, atol=0)
        self.assertEqual(tuple(continuous.time(t).shape), (3, 16))

    def test_paired_run_and_output_protection(self):
        import torch
        from exposure.time_embedding_probe import run

        torch.set_num_threads(1)
        definitions = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]}
        with tempfile.TemporaryDirectory() as directory:
            target, out = Path(directory) / "target.png", Path(directory) / "run"
            Image.new("RGB", (64, 64), "white").save(target)
            kwargs = dict(rule="red", steps=2, batch_size=1, seed=7, init_seed=8,
                          width=4, timesteps=8, lr=0.0001, device="cpu", deterministic=False,
                          eval_times=[0, 7], eval_seeds=[0], sample_seeds=[1])
            report = run(definitions, target, out, **kwargs)
            self.assertEqual(report["arms"]["learned"]["training"]["timestep_counts"],
                             report["arms"]["sinusoidal"]["training"]["timestep_counts"])
            self.assertTrue((out / "learned-1.png").is_file())
            self.assertTrue((out / "sinusoidal-1.png").is_file())
            checkpoint = torch.load(out / "sinusoidal.pt", map_location="cpu", weights_only=True)
            self.assertTrue(checkpoint["diagnostic"]["requires_custom_sampler"])
            with self.assertRaises(ValueError):
                run(definitions, target, out, **kwargs)
