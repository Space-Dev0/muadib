import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional pilot dependency is not installed")
class HighTBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import high_t_baseline, pilot_train
        from exposure.render import render_scene

        cls.torch = torch
        cls.module = high_t_baseline
        torch.set_num_threads(1)
        definitions = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]}
        cls.checkpoint, _ = pilot_train.train(
            definitions, None, steps=1, batch_size=1, seed=3,
            device="cpu", width=4, timesteps=4, lr=1e-4,
            dropout=0.1, ema_decay=0.9, max_attempts=10000,
        )
        cls.target = render_scene(42).image

    def test_oracle_reconstructs_target_and_baselines_are_finite(self):
        torch = self.torch
        report = self.module.diagnose(self.checkpoint, self.target, timesteps=[3], seeds=[0])

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["predictors"], ["raw", "ema", "copy", "zero", "oracle"])
        self.assertEqual(report["timesteps"], [3])
        self.assertEqual(report["seeds"], [0])

        seed_result = report["results"][0]["seeds"][0]
        oracle = seed_result["predictors"]["oracle"]
        self.assertLess(oracle["noise_mse"], 1e-12)
        self.assertLess(oracle["x0_mse"], 1e-10)
        self.assertLess(oracle["x0_clipped_mse"], 1e-10)

        for name in self.module.PREDICTORS:
            metrics = seed_result["predictors"][name]
            for key, value in metrics.items():
                self.assertTrue(torch.isfinite(torch.tensor(value)), (name, key, value))
            self.assertLessEqual(metrics["x0_clipped_mse"], metrics["x0_mse"] + 1e-12)

        self.assertGreater(seed_result["predictors"]["zero"]["noise_mse"], 0.0)
        self.assertGreater(seed_result["predictors"]["copy"]["noise_mse"], 0.0)
        self.assertIn("not a full reverse-trajectory", report["note"])

    def test_matched_noise_reproducibility_and_raw_ema_identity(self):
        matched = copy.deepcopy(self.checkpoint)
        matched["ema"] = copy.deepcopy(matched["model"])
        kwargs = {"timesteps": [1, 3], "seeds": [2, 7]}

        first = self.module.diagnose(matched, self.target, **kwargs)
        second = self.module.diagnose(matched, self.target, **kwargs)
        self.assertEqual(first, second)

        for result in first["results"]:
            for seed_result in result["seeds"]:
                self.assertEqual(seed_result["predictors"]["raw"], seed_result["predictors"]["ema"])

        other = self.module.diagnose(matched, self.target, timesteps=[3], seeds=[8])
        self.assertNotEqual(
            first["results"][-1]["seeds"][0]["predictors"]["copy"],
            other["results"][0]["seeds"][0]["predictors"]["copy"],
        )

    def test_input_validation(self):
        bad_options = (
            {"timesteps": [], "seeds": [0]},
            {"timesteps": [4], "seeds": [0]},
            {"timesteps": [-1], "seeds": [0]},
            {"timesteps": [1, 1], "seeds": [0]},
            {"timesteps": [True], "seeds": [0]},
            {"timesteps": [1], "seeds": []},
            {"timesteps": [1], "seeds": [-1]},
            {"timesteps": [1], "seeds": [1, 1]},
            {"timesteps": [1], "seeds": [True]},
        )
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.module.diagnose(self.checkpoint, self.target, **options)

        with self.assertRaises(ValueError):
            self.module.diagnose(self.checkpoint, self.target, timesteps=[1], seeds=[0], rule="blue")

        bad = self.torch.ones((1, 3, 64, 64)) * 2.0
        with self.assertRaises(ValueError):
            self.module.diagnose(self.checkpoint, bad, timesteps=[1], seeds=[0])

        missing = copy.deepcopy(self.checkpoint)
        del missing["ema"]
        with self.assertRaises(ValueError):
            self.module.diagnose(missing, self.target, timesteps=[1], seeds=[0])

    def test_cli_json_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            target_path = root / "target.png"
            output_path = root / "report.json"
            self.torch.save(self.checkpoint, checkpoint_path)
            self.target.save(target_path)

            checkpoint_before = checkpoint_path.read_bytes()
            target_before = target_path.read_bytes()
            command = [
                sys.executable, "-m", "exposure.high_t_baseline",
                "--checkpoint", str(checkpoint_path),
                "--target", str(target_path),
                "--rule", "red",
                "--timesteps", "3",
                "--seeds", "0", "1",
            ]

            completed = subprocess.run(command + ["--output", str(output_path)],
                                       check=True, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(len(report["results"][0]["seeds"]), 2)

            existing = subprocess.run(command + ["--output", str(output_path)],
                                      capture_output=True, text=True)
            self.assertNotEqual(existing.returncode, 0)

            overwrite_checkpoint = subprocess.run(command + ["--output", str(checkpoint_path)],
                                                  capture_output=True, text=True)
            self.assertNotEqual(overwrite_checkpoint.returncode, 0)
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before)

            overwrite_target = subprocess.run(command + ["--output", str(target_path)],
                                              capture_output=True, text=True)
            self.assertNotEqual(overwrite_target.returncode, 0)
            self.assertEqual(target_path.read_bytes(), target_before)


if __name__ == "__main__":
    unittest.main()
