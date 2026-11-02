import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional pilot dependency is not installed")
class IdentityEpsFinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import identity_eps_finetune

        cls.torch = torch
        cls.module = identity_eps_finetune
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(5)
        cls.target_tensor = torch.rand((1, 3, 64, 64), generator=generator) * 2.0 - 1.0
        cls.definitions = {
            "red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}],
            "blue": [{"type": "attribute", "variable": "a", "field": "color", "value": "blue"}],
        }
        cls.base_kwargs = dict(
            rule="red",
            steps=1,
            batch_size=1,
            seed=0,
            init_seed=0,
            width=4,
            timesteps=8,
            lr=1e-4,
            dropout=0.0,
            ema_decay=0.9,
            device="cpu",
            deterministic=False,
            eval_timesteps=[0, 3, 7],
            eval_seeds=[0, 1],
            sample_seeds=[0],
        )

    def test_predict_noise_modes_use_the_same_backbone(self):
        torch = self.torch

        class Backbone(torch.nn.Module):
            def forward(self, x, t, tokens, drop=None):
                return torch.full_like(x, 0.25)

        noisy = torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 2, 2)
        time = torch.tensor([3])
        tokens = torch.zeros((1, 1, 4), dtype=torch.long)
        backbone = Backbone()

        standard = self.module.predict_noise(backbone, noisy, time, tokens, None, "standard")
        identity = self.module.predict_noise(backbone, noisy, time, tokens, None, "identity_residual")
        torch.testing.assert_close(standard, torch.full_like(noisy, 0.25))
        torch.testing.assert_close(identity, noisy + 0.25)

        with self.assertRaises(ValueError):
            self.module.predict_noise(backbone, noisy, time, tokens, None, "unknown")

        zero = lambda x, t, tokens, drop=None: torch.zeros_like(x)
        copy = self.module.predict_noise(zero, noisy, time, tokens, None, "identity_residual")
        torch.testing.assert_close(copy, noisy)

    def test_matched_initialization_hash_is_reproducible(self):
        torch = self.torch
        first = self.module._build_model(4, 8, torch.device("cpu"), 46)
        second = self.module._build_model(4, 8, torch.device("cpu"), 46)
        different = self.module._build_model(4, 8, torch.device("cpu"), 47)

        first_hash = self.module._model_hash(first)
        self.assertEqual(first_hash, self.module._model_hash(second))
        self.assertNotEqual(first_hash, self.module._model_hash(different))

    def test_paired_experiment_records_matched_arms_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            report = self.module.run_experiment(
                self.definitions,
                self.target_tensor,
                output_dir=output,
                target_path=Path(directory) / "target.png",
                **self.base_kwargs,
            )

            self.assertEqual(report["experiment"], "paired_fixed_scene_identity_eps")
            self.assertEqual(report["schedule_timesteps"], 8)
            self.assertEqual(report["config"]["timesteps"], 8)
            self.assertGreater(report["terminal_alpha_bar"], 0.01)
            self.assertTrue(report["prior_approximation_warning"])
            self.assertGreater(report["terminal_epsilon_to_x0_amplification"], 0.0)
            self.assertGreater(report["baselines"]["constant_target_mean_mse_0_1"], 0.0)
            self.assertEqual(len(report["baselines"]["target_mean_rgb_0_1"]), 3)
            self.assertEqual(len(report["baselines"]["identity_epsilon_sampler"]), 1)
            self.assertTrue(math.isfinite(
                report["baselines"]["identity_epsilon_sampler"][0]["mse_0_1"]
            ))
            self.assertEqual(report["rule"], "red")
            self.assertEqual(set(report["arms"]), {"standard", "identity_residual"})
            self.assertTrue(report["matched"]["same_initialization"])
            self.assertTrue(report["matched"]["same_realized_timesteps"])
            self.assertTrue(report["matched"]["shared_draw_streams"])
            self.assertEqual(report["matched"]["full_schedule"], 8)
            self.assertEqual(
                report["matched"]["generator_seeds"],
                {"time": 0, "noise": 7919, "dropout": 104729},
            )
            hashes = report["matched"]["initial_parameter_sha256"]
            self.assertEqual(hashes["standard"], hashes["identity_residual"])

            for mode in ("standard", "identity_residual"):
                arm = report["arms"][mode]
                self.assertEqual(arm["mode"], mode)
                self.assertEqual(arm["training"]["mode"], mode)
                self.assertEqual(arm["training"]["steps"], 1)
                self.assertEqual(len(arm["training"]["loss_history"]), 1)
                self.assertTrue(math.isfinite(arm["training"]["last_loss"]))
                for split in ("raw", "ema"):
                    results = arm["metrics"][split]["results"]
                    self.assertEqual([row["timestep"] for row in results], [0, 3, 7])
                    for row in results:
                        self.assertEqual(set(row["averages"]), {"model", "copy", "zero", "oracle"})
                self.assertEqual(len(arm["samples"]), 1)
                self.assertTrue(math.isfinite(arm["samples"][0]["mse_0_1"]))

            self.assertEqual(
                report["arms"]["standard"]["training"]["timestep_counts"],
                report["arms"]["identity_residual"]["training"]["timestep_counts"],
            )
            self.assertEqual(
                set(report["comparison"]),
                {"raw_identity_minus_standard", "ema_identity_minus_standard"},
            )
            comparison = report["comparison"]["ema_identity_minus_standard"]["rows"]
            self.assertEqual(set(comparison), {"0", "3", "7"})

            self.assertTrue((output / "standard.pt").is_file())
            self.assertTrue((output / "identity_residual.pt").is_file())
            self.assertTrue((output / "samples" / "standard-0.png").is_file())
            self.assertTrue((output / "samples" / "identity_residual-0.png").is_file())
            self.assertTrue((output / "samples" / "identity-epsilon-0.png").is_file())
            self.assertTrue((output / "report.json").is_file())

            saved = self.torch.load(output / "standard.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["schema_version"], 0)
            self.assertEqual(saved["diagnostic"]["kind"], "identity_eps_finetune_arm")
            self.assertFalse(saved["diagnostic"]["resumable"])
            self.assertFalse(saved["diagnostic"]["requires_custom_sampler"])
            saved_identity = self.torch.load(
                output / "identity_residual.pt", map_location="cpu", weights_only=True
            )
            self.assertTrue(saved_identity["diagnostic"]["requires_custom_sampler"])

            on_disk = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["experiment"], report["experiment"])
            self.assertEqual(
                set(on_disk["arms"]["standard"]["metrics"]["ema"]["results"][0]["averages"]),
                {"model", "copy", "zero", "oracle"},
            )

    def test_validation_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(ValueError):
                self.module.run_experiment(
                    self.definitions, self.target_tensor, output_dir=existing, **self.base_kwargs
                )

            invalid_options = (
                {"steps": 0},
                {"batch_size": 0},
                {"seed": -1},
                {"init_seed": -1},
                {"width": 5},
                {"timesteps": 1},
                {"lr": 0.0},
                {"lr": float("inf")},
                {"dropout": 1.0},
                {"ema_decay": -0.1},
                {"deterministic": 1},
                {"device": "tpu"},
                {"rule": "green"},
                {"eval_timesteps": [1, 1]},
                {"eval_timesteps": [8]},
                {"eval_seeds": [-1]},
                {"eval_seeds": [1, 1]},
                {"sample_seeds": []},
                {"sample_seeds": [2, 2]},
            )
            for index, options in enumerate(invalid_options):
                with self.subTest(options=options):
                    target = root / f"output-{index}"
                    with self.assertRaises(ValueError):
                        self.module.run_experiment(
                            self.definitions,
                            self.target_tensor,
                            output_dir=target,
                            **{**self.base_kwargs, **options},
                        )
                    self.assertFalse(target.exists())

            with self.assertRaises(ValueError):
                self.module.run_experiment(
                    {},
                    self.target_tensor,
                    output_dir=root / "empty-rules",
                    **{**self.base_kwargs, "rule": None},
                )

            with self.assertRaises(ValueError):
                self.module.run_experiment(
                    self.definitions,
                    self.target_tensor,
                    output_dir=root / "ambiguous-rule",
                    **{**self.base_kwargs, "rule": None},
                )

            bad_targets = (
                self.torch.zeros((1, 3, 32, 32)),
                self.torch.full((1, 3, 64, 64), 2.0),
                self.torch.full((1, 3, 64, 64), float("nan")),
            )
            for index, target in enumerate(bad_targets):
                with self.subTest(target=index):
                    with self.assertRaises(ValueError):
                        self.module.run_experiment(
                            self.definitions,
                            target,
                            output_dir=root / f"bad-target-{index}",
                            **self.base_kwargs,
                        )

    def test_cli_writes_report_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rules_path = root / "rules.json"
            target_path = root / "target.png"
            output_dir = root / "run"
            rules_path.write_text(json.dumps(self.definitions), encoding="utf-8")
            pixels = ((self.target_tensor[0] + 1.0) * 127.5).round().to(self.torch.uint8)
            Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode="RGB").save(target_path)
            rules_before = rules_path.read_bytes()
            target_before = target_path.read_bytes()

            command = [
                sys.executable, "-m", "exposure.identity_eps_finetune",
                "--rules", str(rules_path),
                "--target", str(target_path),
                "--output-dir", str(output_dir),
                "--rule", "red",
                "--steps", "1",
                "--batch-size", "1",
                "--seed", "3",
                "--init-seed", "3",
                "--width", "4",
                "--timesteps", "8",
                "--device", "cpu",
                "--eval-timesteps", "2", "7",
                "--eval-seeds", "0",
                "--sample-seeds", "0",
            ]
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            self.assertEqual(completed.returncode, 0)
            report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rules"], str(rules_path))
            self.assertEqual(report["target"], str(target_path))
            self.assertIn("rules_sha256", report)
            self.assertIn("target_sha256", report)
            self.assertEqual(report["config"]["seed"], 3)
            self.assertEqual(report["config"]["init_seed"], 3)
            self.assertTrue((output_dir / "standard.pt").is_file())
            self.assertTrue((output_dir / "identity_residual.pt").is_file())
            self.assertTrue((output_dir / "samples" / "standard-0.png").is_file())
            self.assertTrue((output_dir / "samples" / "identity_residual-0.png").is_file())
            self.assertTrue((output_dir / "samples" / "identity-epsilon-0.png").is_file())

            failed = subprocess.run(
                command, capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(rules_path.read_bytes(), rules_before)
            self.assertEqual(target_path.read_bytes(), target_before)


if __name__ == "__main__":
    unittest.main()
