import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional pilot dependency is not installed")
class HighTFinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import high_t_finetune, pilot_train
        from exposure.render import render_scene

        cls.torch = torch
        cls.module = high_t_finetune
        torch.set_num_threads(1)
        definitions = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]}
        cls.checkpoint, _ = pilot_train.train(
            definitions, None, steps=1, batch_size=1, seed=3,
            device="cpu", width=4, timesteps=8, lr=1e-4,
            dropout=0.1, ema_decay=0.9, max_attempts=10000,
        )
        cls.target = render_scene(42).image

    def test_loss_weights_are_normalized_and_only_change_high_timesteps(self):
        torch = self.torch
        timesteps = torch.tensor([0, 5, 6, 7])
        weights = self.module.loss_weights(
            timesteps, high_t_min=6, high_t_weight=5.0, total_timesteps=8
        )
        expected = torch.tensor([0.5, 0.5, 2.5, 2.5])
        torch.testing.assert_close(weights, expected)
        full_schedule = self.module.loss_weights(
            torch.arange(8), high_t_min=6, high_t_weight=5.0, total_timesteps=8
        )
        self.assertAlmostEqual(float(full_schedule.mean()), 1.0, places=6)

        baseline = self.module.loss_weights(
            timesteps, high_t_min=6, high_t_weight=1.0, total_timesteps=8
        )
        torch.testing.assert_close(baseline, torch.ones_like(baseline))

        for options in ({"high_t_min": -1, "high_t_weight": 5.0, "total_timesteps": 8},
                        {"high_t_min": 6, "high_t_weight": 0.5, "total_timesteps": 8},
                        {"high_t_min": 6, "high_t_weight": float("nan"), "total_timesteps": 8},
                        {"high_t_min": 8, "high_t_weight": 5.0, "total_timesteps": 8},
                        {"high_t_min": 6, "high_t_weight": 5.0, "total_timesteps": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.module.loss_weights(timesteps, **options)

    def test_paired_arms_share_timesteps_and_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            report = self.module.run_experiment(
                self.checkpoint, self.target, output_dir=output, rule="red",
                steps=3, batch_size=2, seed=11, device="cpu", high_t_min=6,
                high_t_weight=5.0, eval_timesteps=[0, 2, 7], eval_seeds=[0, 1],
            )

            self.assertEqual(report["experiment"], "paired_high_noise_finetune")
            self.assertEqual(report["schedule_timesteps"], 8)
            self.assertTrue(report["matched"]["noise_and_dropout_masks"])
            self.assertEqual(set(report["arms"]), {"baseline", "weighted"})
            self.assertEqual(report["arms"]["baseline"]["high_t_weight"], 1.0)
            self.assertEqual(report["arms"]["weighted"]["high_t_weight"], 5.0)
            self.assertEqual(
                report["arms"]["baseline"]["training"]["timestep_counts"],
                report["arms"]["weighted"]["training"]["timestep_counts"],
            )
            self.assertEqual(
                report["arms"]["baseline"]["training"]["unweighted_mse_history"][0],
                report["arms"]["weighted"]["training"]["unweighted_mse_history"][0],
            )
            self.assertIn("weighted_minus_baseline", report["comparison"])
            self.assertAlmostEqual(report["objective_weights"]["expected_weight"], 2.0)
            self.assertAlmostEqual(report["objective_weights"]["normalized_low_t_weight"], 0.5)
            self.assertAlmostEqual(report["objective_weights"]["normalized_high_t_weight"], 2.5)
            self.assertEqual(len(report["start_metrics"]["results"]), 3)
            self.assertEqual(len(report["arms"]["baseline"]["metrics"]["results"]), 3)
            self.assertEqual(len(report["arms"]["weighted"]["metrics"]["results"]), 3)

            self.assertTrue((output / "baseline.pt").is_file())
            self.assertTrue((output / "weighted.pt").is_file())
            self.assertTrue((output / "report.json").is_file())
            self.assertIn("baseline_checkpoint", report["outputs"])

    def test_input_validation_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            base = dict(output_dir=existing, rule="red", steps=1, batch_size=1,
                        seed=0, device="cpu", high_t_min=6, high_t_weight=5.0,
                        eval_timesteps=[0, 7], eval_seeds=[0])
            with self.assertRaises(ValueError):
                self.module.run_experiment(self.checkpoint, self.target, **base)

            missing = copy.deepcopy(self.checkpoint)
            del missing["ema"]
            with self.assertRaises(ValueError):
                self.module.run_experiment(missing, self.target, **{**base, "output_dir": Path(directory) / "bad"})

            for options in (
                {"steps": 0},
                {"batch_size": 0},
                {"seed": -1},
                {"high_t_min": 8},
                {"high_t_weight": 1.0},
                {"eval_timesteps": [1, 1]},
                {"eval_timesteps": [8]},
                {"eval_seeds": [-1]},
                {"eval_seeds": [1, 1]},
                {"device": "tpu"},
                {"rule": "blue"},
            ):
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):
                        self.module.run_experiment(
                            self.checkpoint, self.target,
                            **{**base, "output_dir": Path(directory) / "bad", **options},
                        )

    def test_cli_writes_report_and_protects_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            target_path = root / "target.png"
            output_dir = root / "run"
            self.torch.save(self.checkpoint, checkpoint_path)
            self.target.save(target_path)
            checkpoint_before = checkpoint_path.read_bytes()
            target_before = target_path.read_bytes()

            command = [
                sys.executable, "-m", "exposure.high_t_finetune",
                "--checkpoint", str(checkpoint_path),
                "--target", str(target_path),
                "--output-dir", str(output_dir),
                "--rule", "red",
                "--steps", "1",
                "--batch-size", "1",
                "--seed", "5",
                "--device", "cpu",
                "--high-t-min", "6",
                "--high-t-weight", "3",
                "--eval-timesteps", "2", "7",
                "--eval-seeds", "0",
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0)
            report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["checkpoint"], str(checkpoint_path))
            self.assertEqual(report["target"], str(target_path))
            self.assertIn("checkpoint_sha256", report)
            self.assertTrue((output_dir / "baseline.pt").is_file())
            self.assertTrue((output_dir / "weighted.pt").is_file())

            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before)
            self.assertEqual(target_path.read_bytes(), target_before)


if __name__ == "__main__":
    unittest.main()
