import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency unavailable")
class OptimizerProbeTests(unittest.TestCase):
    def test_paired_training_and_custom_checkpoint(self):
        import torch
        from exposure import optimizer_probe, pilot_train
        torch.set_num_threads(1)
        definitions = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.png"
            Image.new("RGB", (64, 64), (255, 0, 0)).save(target)
            output = Path(directory) / "result"
            options = dict(rule="red", steps=2, batch_size=2, seed=4, init_seed=9, width=4,
                           timesteps=8, adamw_lr=1e-4, sgd_lr=0.01, device="cpu", deterministic=False,
                           eval_times=[0, 7], eval_seeds=[0], sample_seeds=[1])
            report = optimizer_probe.run(definitions, target, output, **options)
            self.assertEqual(report["initial_model_sha256"]["adamw"], report["initial_model_sha256"]["sgd"])
            self.assertEqual(report["arms"]["adamw"]["timestep_counts"], report["arms"]["sgd"]["timestep_counts"])
            self.assertEqual(report["config"]["sgd_lr"], 0.01)
            for name in ("adamw", "sgd"):
                checkpoint = torch.load(output / f"{name}.pt", map_location="cpu", weights_only=True)
                with self.assertRaises(ValueError):
                    pilot_train.sample(checkpoint, "red", seed=1, device="cpu", guidance=1)
                self.assertTrue((output / f"{name}-1.png").is_file())
            with self.assertRaises(ValueError):
                optimizer_probe.run(definitions, target, output, **options)

    def test_bad_learning_rate(self):
        from exposure import optimizer_probe
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                optimizer_probe.run({}, "target.png", Path(directory) / "out", rule="red", steps=1,
                    batch_size=1, seed=0, init_seed=0, width=4, timesteps=8, adamw_lr=1e-4,
                    sgd_lr=float("nan"), device="cpu", deterministic=False, eval_times=[0],
                    eval_seeds=[0], sample_seeds=[0])


if __name__ == "__main__":
    unittest.main()
