import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional pilot dependency is not installed")
class DenoiseProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import denoise_probe, pilot_train
        cls.torch, cls.probe = torch, denoise_probe
        torch.set_num_threads(1)
        definitions = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]}
        cls.checkpoint, _ = pilot_train.train(definitions, None, steps=1, batch_size=1, seed=3,
                                               device="cpu", width=4, timesteps=3, lr=1e-4,
                                               dropout=0.1, ema_decay=0.9, max_attempts=10000)

    def test_oracle_noise_recovers_clean_input(self):
        torch = self.torch
        clean = torch.tensor([[-1.0, 0.5, 1.0]])
        noise = torch.tensor([[0.25, -0.75, 0.1]])
        abar = torch.tensor(0.4)
        noisy = abar.sqrt() * clean + (1 - abar).sqrt() * noise
        torch.testing.assert_close(self.probe.reconstruct_x0(noisy, noise, abar), clean)

    def test_matched_inputs_and_seed_are_reproducible(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["ema"] = copy.deepcopy(checkpoint["model"])
        kwargs = {"rule": "red", "samples": 2, "timesteps": [0, 2], "seed": 7}
        result = self.probe.diagnose(checkpoint, **kwargs)
        self.assertEqual(result, self.probe.diagnose(checkpoint, **kwargs))
        for cell in result["results"]:
            self.assertEqual(cell["metrics"]["model"], cell["metrics"]["ema"])
            self.assertGreater(cell["metrics"]["zero"]["noise_mse"], 0)
        for options in ({"samples": 0}, {"seed": -1}, {"timesteps": [1, 1]},
                        {"timesteps": [3]}, {"rule": "missing"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.probe.diagnose(checkpoint, **{**kwargs, **options})

    def test_cli_json_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, report = root / "pilot.pt", root / "report.json"
            self.torch.save(self.checkpoint, checkpoint)
            command = [sys.executable, "-m", "exposure.denoise_probe", "--checkpoint", str(checkpoint),
                       "--rule", "red", "--samples", "1", "--timesteps", "0", "2", "--seed", "4"]
            subprocess.run(command + ["--output", str(report)], check=True, capture_output=True, text=True)
            self.assertEqual(len(json.loads(report.read_text())["results"]), 2)
            saved = checkpoint.read_bytes()
            failed = subprocess.run(command + ["--output", str(checkpoint)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(checkpoint.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
