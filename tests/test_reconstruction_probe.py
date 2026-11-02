import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional pilot dependency is not installed")
class ReconstructionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import pilot_train, reconstruction_probe
        from exposure.render import render_scene

        cls.torch = torch
        cls.pilot = pilot_train
        cls.probe = reconstruction_probe
        torch.set_num_threads(1)

        cls.definitions = {
            "red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}],
            "blue": [{"type": "attribute", "variable": "a", "field": "color", "value": "blue"}],
        }

        # Train a 1-step checkpoint with full 1000-step schedule on CPU
        cls.checkpoint, _ = pilot_train.train(
            cls.definitions, None, steps=1, batch_size=1, seed=3,
            device="cpu", width=4, timesteps=1000, lr=1e-4,
            dropout=0.1, ema_decay=0.9, max_attempts=10000
        )

        # Create a static-scene checkpoint (schema_version 0, single rule)
        cls.static_checkpoint = copy.deepcopy(cls.checkpoint)
        cls.static_checkpoint["schema_version"] = 0
        cls.static_checkpoint["rules"] = {"red": cls.definitions["red"]}
        cls.static_checkpoint["diagnostic"] = {
            "kind": "single_fixed_render", "scene_seed": 77, "target_attempts": 1, "resumable": False
        }

        # Render a known deterministic target image
        cls.target_image = render_scene(42).image

    def test_exact_noise_oracle_reconstructs_known_image(self):
        torch, probe = self.torch, self.probe

        pixels = np.array(self.target_image, dtype=np.float32, copy=True)
        clean = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        beta, alpha_bar = probe.schedule(1000, torch.device("cpu"))

        class Oracle(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.register_buffer("clean", clean, persistent=False)
                self.register_buffer("alpha_bar", alpha_bar, persistent=False)

            def forward(self, x, timestep, tokens=None, drop=None):
                abar = self.alpha_bar[timestep, None, None, None]
                return (x - abar.sqrt() * self.clean) / (1.0 - abar).sqrt()

            def load_state_dict(self, state_dict, strict=True):
                pass

        oracle = Oracle()
        tokens = torch.zeros((1, 1, 4), dtype=torch.long)

        # Test reverse_trajectory across multiple timesteps
        for start_step in (10, 50, 500):
            gen = torch.Generator(device="cpu").manual_seed(123)
            starting_noise = torch.randn(clean.shape, generator=gen, device="cpu")
            abar = alpha_bar[start_step]
            noisy_start = abar.sqrt() * clean + (1.0 - abar).sqrt() * starting_noise
            transition_noises = {
                step: torch.randn(clean.shape, generator=gen, device="cpu")
                for step in range(start_step, 0, -1)
            }

            reconstructed = probe.reverse_trajectory(
                oracle, noisy_start, tokens, start_step, beta, alpha_bar, transition_noises
            )
            torch.testing.assert_close(reconstructed, clean, atol=1e-5, rtol=1e-5)
            mse = float((reconstructed - clean).square().mean())
            self.assertLess(mse, 1e-9)

        # Also test diagnose pipeline with Oracle
        with patch.object(probe, "PilotUNet", Oracle):
            res = probe.diagnose(
                self.checkpoint, self.target_image, timesteps=[20, 100], seeds=[0], rule="red"
            )
        for entry in res["results"]:
            seed_res = entry["seeds"][0]
            self.assertGreater(seed_res["baseline_input_mse"], 0.0)
            self.assertLess(seed_res["raw"]["reconstruction_mse"], 1e-9)
            self.assertLess(seed_res["ema"]["reconstruction_mse"], 1e-9)

    def test_raw_and_ema_use_matched_noise(self):
        probe = self.probe

        # When raw and EMA weights are identical, results must match bitwise
        matched_checkpoint = copy.deepcopy(self.checkpoint)
        matched_checkpoint["ema"] = copy.deepcopy(matched_checkpoint["model"])

        res = probe.diagnose(
            matched_checkpoint, self.target_image, timesteps=[20, 100], seeds=[0, 1], rule="red"
        )
        for entry in res["results"]:
            for seed_res in entry["seeds"]:
                self.assertEqual(seed_res["raw"]["reconstruction_mse"], seed_res["ema"]["reconstruction_mse"])
                self.assertGreater(seed_res["baseline_input_mse"], 0.0)

        # When weights differ, baseline input MSE is still identical for both
        diff_res = probe.diagnose(
            self.checkpoint, self.target_image, timesteps=[20], seeds=[0], rule="red"
        )
        seed_res = diff_res["results"][0]["seeds"][0]
        # Both share the same baseline_input_mse
        self.assertEqual(seed_res["baseline_input_mse"], res["results"][0]["seeds"][0]["baseline_input_mse"])

    def test_seed_reproducibility(self):
        probe = self.probe

        # Same seeds must produce identical results
        run1 = probe.diagnose(
            self.checkpoint, self.target_image, timesteps=[20, 100], seeds=[42, 43], rule="red"
        )
        run2 = probe.diagnose(
            self.checkpoint, self.target_image, timesteps=[20, 100], seeds=[42, 43], rule="red"
        )
        self.assertEqual(run1, run2)

        # Different seeds must produce different noise and baseline input MSE
        seed0_base = run1["results"][0]["seeds"][0]["baseline_input_mse"]
        seed1_base = run1["results"][0]["seeds"][1]["baseline_input_mse"]
        self.assertNotEqual(seed0_base, seed1_base)

    def test_cli_json_and_input_output_protection(self):
        torch = self.torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt_path = root / "checkpoint.pt"
            target_path = root / "target.png"
            report_path = root / "report.json"

            torch.save(self.checkpoint, ckpt_path)
            self.target_image.save(target_path)

            ckpt_bytes_before = ckpt_path.read_bytes()
            target_bytes_before = target_path.read_bytes()

            cmd = [
                sys.executable, "-m", "exposure.reconstruction_probe",
                "--checkpoint", str(ckpt_path),
                "--target", str(target_path),
                "--rule", "red",
                "--timesteps", "20", "100",
                "--seeds", "0", "1",
                "--output", str(report_path),
            ]

            # Normal successful run
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0)
            self.assertTrue(report_path.is_file())

            # Parse and verify written report
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["schedule_timesteps"], 1000)
            self.assertEqual(report["timesteps"], [20, 100])
            self.assertEqual(report["seeds"], [0, 1])
            self.assertEqual(len(report["results"]), 2)
            for res_entry in report["results"]:
                self.assertEqual(len(res_entry["seeds"]), 2)
                self.assertIn("baseline_input_mse", res_entry["averages"])
                self.assertIn("raw", res_entry["averages"])
                self.assertIn("ema", res_entry["averages"])
            self.assertIn("reverse loop is an instrument check", report["note"])

            # Refuse to overwrite existing report file
            failed_existing = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(failed_existing.returncode, 0)

            # Refuse to overwrite checkpoint
            cmd_ckpt_out = list(cmd)
            cmd_ckpt_out[-1] = str(ckpt_path)
            failed_ckpt = subprocess.run(cmd_ckpt_out, capture_output=True, text=True)
            self.assertNotEqual(failed_ckpt.returncode, 0)

            # Refuse to overwrite target image
            cmd_target_out = list(cmd)
            cmd_target_out[-1] = str(target_path)
            failed_target = subprocess.run(cmd_target_out, capture_output=True, text=True)
            self.assertNotEqual(failed_target.returncode, 0)

            # Check that input files remain completely intact and protected
            self.assertEqual(ckpt_path.read_bytes(), ckpt_bytes_before)
            self.assertEqual(target_path.read_bytes(), target_bytes_before)

    def test_static_scene_checkpoint_and_auto_rule(self):
        probe = self.probe

        # Static checkpoint has 1 rule, so rule argument can be omitted
        res = probe.diagnose(
            self.static_checkpoint, self.target_image, timesteps=[20], seeds=[0]
        )
        self.assertEqual(res["rule"], "red")
        self.assertEqual(res["schedule_timesteps"], 1000)
        self.assertEqual(len(res["results"]), 1)

    def test_target_input_types(self):
        torch, probe = self.torch, self.probe

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.png"
            self.target_image.save(path)

            # Path
            res_path = probe.diagnose(
                self.static_checkpoint, path, timesteps=[20], seeds=[0]
            )

            # PIL Image
            res_img = probe.diagnose(
                self.static_checkpoint, self.target_image, timesteps=[20], seeds=[0]
            )

            # Tensor
            clean = probe.load_target(self.target_image)[0]
            res_tensor = probe.diagnose(
                self.static_checkpoint, clean, timesteps=[20], seeds=[0]
            )

            # All three target input types give identical results
            self.assertEqual(
                res_path["results"][0]["seeds"][0]["raw"]["reconstruction_mse"],
                res_img["results"][0]["seeds"][0]["raw"]["reconstruction_mse"]
            )
            self.assertEqual(
                res_path["results"][0]["seeds"][0]["raw"]["reconstruction_mse"],
                res_tensor["results"][0]["seeds"][0]["raw"]["reconstruction_mse"]
            )

    def test_validation_errors(self):
        probe = self.probe
        valid_kwargs = {
            "checkpoint": self.checkpoint,
            "target": self.target_image,
            "timesteps": [20, 100],
            "seeds": [0],
            "rule": "red",
        }

        # Timestep out of bounds
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "timesteps": [1000]})
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "timesteps": [-1]})

        # Duplicate timesteps
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "timesteps": [20, 20]})

        # Empty timesteps
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "timesteps": []})

        # Negative seed
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "seeds": [-1]})

        # Empty seeds
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "seeds": []})

        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "seeds": [0, 0]})

        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "target": self.torch.full((3, 64, 64), 2.0)})

        # Missing rule when checkpoint has multiple rules
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "rule": None})

        # Unknown rule
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "rule": "nonexistent"})

        # Wrong image size
        bad_image = Image.new("RGB", (32, 32))
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "target": bad_image})

        # Non-dict checkpoint
        with self.assertRaises(TypeError):
            probe.diagnose(**{**valid_kwargs, "checkpoint": "not_a_dict"})

        # Missing keys in checkpoint
        bad_checkpoint = copy.deepcopy(self.checkpoint)
        del bad_checkpoint["ema"]
        with self.assertRaises(ValueError):
            probe.diagnose(**{**valid_kwargs, "checkpoint": bad_checkpoint})

    def test_no_arbitrary_thresholds_or_fabricated_labels(self):
        probe = self.probe

        res = probe.diagnose(
            self.checkpoint, self.target_image, timesteps=[20], seeds=[0], rule="red"
        )
        report_str = json.dumps(res)

        # Ensure no fabricated qualitative human labels
        forbidden = ["\"pass\"", "\"fail\"", "\"passed\"", "\"failed\"", "\"grade\"", "\"verdict\""]
        for term in forbidden:
            self.assertNotIn(term, report_str.lower())

        # Ensure required disclaimer note is present
        self.assertIn("Truncated reconstruction diagnostic", res["note"])
        self.assertIn("is not proof of sampler correctness, population quality or Stage 0 passage", res["note"])


if __name__ == "__main__":
    unittest.main()
