import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional pilot dependency is not installed")
class PilotTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import pilot_train
        cls.torch, cls.pilot = torch, pilot_train
        torch.set_num_threads(1)
        cls.definitions = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}],
                           "blue_above": [{"type": "attribute", "variable": "a", "field": "color", "value": "blue"},
                                          {"type": "relation", "subject": "a", "kind": "above", "object": "b"}]}

    def test_token_structure_and_condition_null_forward(self):
        torch, pilot = self.torch, self.pilot
        rules = pilot.load_rules(self.definitions)
        tokens = pilot.encode_rules(rules)
        self.assertEqual(tuple(tokens.shape), (2, 2, 4))
        self.assertEqual(tokens[1, 1].tolist(), [4, 1, 2, 0])
        model = pilot.PilotUNet(width=4, timesteps=2)
        images = torch.randn(2, 3, 64, 64)
        out = model(images, torch.tensor([0, 1]), tokens, torch.tensor([False, True]))
        self.assertEqual(tuple(out.shape), (2, 3, 64, 64))
        self.assertTrue(torch.isfinite(out).all())
        out.mean().backward()
        self.assertTrue(torch.isfinite(model.input.weight.grad).all())

    def test_train_checkpoint_reload_and_seeded_sample(self):
        torch, pilot = self.torch, self.pilot
        weights = {"red": 1.0, "blue_above": 0.0}
        options = dict(steps=1, batch_size=1, seed=7, device="cpu", width=4, timesteps=2,
                       lr=1e-4, dropout=0.1, ema_decay=0.9, max_attempts=10000)
        checkpoint, report = pilot.train(self.definitions, weights, **options)
        second, _ = pilot.train(self.definitions, weights, **options)
        self.assertTrue(report["prior_approximation_warning"])
        self.assertEqual(report["condition_counts"], {"red": 1, "blue_above": 0})
        self.assertEqual(report["realized_rule_counts_by_condition"]["red"]["red"], 1)
        self.assertTrue(all(torch.equal(checkpoint["model"][k], second["model"][k])
                            for k in checkpoint["model"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.pt"
            pilot.save_checkpoint(path, checkpoint)
            with self.assertRaises(ValueError):
                pilot.save_checkpoint(path, checkpoint)
            failed_path = Path(directory) / "failed.pt"
            with patch("shutil.copyfileobj", side_effect=OSError("copy failure")):
                with self.assertRaises(OSError):
                    pilot.save_checkpoint(failed_path, checkpoint)
            self.assertFalse(failed_path.exists())
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            first = pilot.sample(loaded, "red", seed=5, device="cpu", guidance=1)
            other = pilot.sample(loaded, "red", seed=5, device="cpu", guidance=1)
            self.assertEqual(first.size, (64, 64))
            self.assertEqual(first.tobytes(), other.tobytes())
            for version in (1, 2):
                legacy = copy.deepcopy(loaded)
                legacy["schema_version"] = version
                del legacy["config"]["deterministic"]
                self.assertEqual(pilot.sample(legacy, "red", seed=5, device="cpu", guidance=1).size, (64, 64))

    def test_invalid_weights_and_sampling_failure_abort(self):
        pilot = self.pilot
        _, normal_schedule = pilot.schedule(1000, self.torch.device("cpu"))
        self.assertLess(float(normal_schedule[-1]), 0.01)
        with self.assertRaises(ValueError):
            pilot._weights(["red"], {"red": 0.5})
        with self.assertRaises(ValueError):
            pilot._weights(["red"], {"red": float("nan")})
        impossible = {"impossible": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"},
                                     {"type": "attribute", "variable": "a", "field": "color", "value": "blue"}]}
        with self.assertRaisesRegex(ValueError, "Condition 'impossible'"):
            pilot.train(impossible, None, steps=1, batch_size=1, seed=0, device="cpu", width=4,
                        timesteps=2, lr=1e-4, dropout=0.1, ema_decay=0.9, max_attempts=2)

    def test_resume_matches_uninterrupted_training(self):
        torch, pilot = self.torch, self.pilot
        options = dict(batch_size=2, seed=22, device="cpu", width=4, timesteps=3,
                       lr=1e-4, dropout=0.1, ema_decay=0.9, max_attempts=10000, deterministic=True)
        weights = {"red": 0.5, "blue_above": 0.5}
        full, full_report = pilot.train(self.definitions, weights, steps=4, **options)
        first, _ = pilot.train(self.definitions, weights, steps=2, **options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "first.pt"
            pilot.save_checkpoint(path, first)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            before_optimizer = copy.deepcopy(restored["optimizer"])
            resumed, resumed_report = pilot.train(self.definitions, weights, steps=2, resume=restored, **options)
            replay, _ = pilot.train(self.definitions, weights, steps=2, resume=restored, **options)
        self.assertEqual(resumed["schema_version"], 3)
        self.assertEqual(resumed["config"]["steps"], 4)
        self.assertTrue(resumed["config"]["deterministic"])
        self.assertTrue(resumed["rng"]["runtime"]["deterministic_algorithms"])
        image = pilot.sample(resumed, "red", seed=5, device="cpu", guidance=1)
        self.assertEqual(image.size, (64, 64))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        for key in ("condition_counts", "realized_rule_counts_by_condition", "sampling_attempts", "last_loss"):
            self.assertEqual(full_report[key], resumed_report[key])
        for state_name in ("model", "ema"):
            for name in full[state_name]:
                self.assertTrue(torch.equal(full[state_name][name], resumed[state_name][name]), name)
                self.assertTrue(torch.equal(resumed[state_name][name], replay[state_name][name]), name)
        self.assertEqual(full["rng"]["python"], resumed["rng"]["python"])
        self.assertTrue(torch.equal(full["rng"]["torch_cpu"], resumed["rng"]["torch_cpu"]))
        self.assertIsNone(resumed["rng"]["torch_cuda"])
        self.assertEqual(full["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
        for key, state in full["optimizer"]["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, resumed["optimizer"]["state"][key][name]), name)
                self.assertTrue(torch.equal(value, replay["optimizer"]["state"][key][name]), name)
                self.assertTrue(torch.equal(before_optimizer["state"][key][name],
                                            restored["optimizer"]["state"][key][name]), name)

    def test_resume_rejects_older_incompatible_and_damaged_checkpoints(self):
        pilot = self.pilot
        options = dict(steps=1, batch_size=1, seed=7, device="cpu", width=4, timesteps=2,
                       lr=1e-4, dropout=0.1, ema_decay=0.9, max_attempts=10000)
        weights = {"red": 1.0, "blue_above": 0.0}
        first, _ = pilot.train(self.definitions, weights, **options)
        old = dict(first, schema_version=2)
        with self.assertRaisesRegex(ValueError, "version 3"):
            pilot.train(self.definitions, weights, resume=old, **options)
        missing = copy.deepcopy(first)
        del missing["rng"]["torch_cpu"]
        with self.assertRaisesRegex(ValueError, "resume state"):
            pilot.train(self.definitions, weights, resume=missing, **options)
        changed = dict(options, batch_size=2)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            pilot.train(self.definitions, weights, resume=first, **changed)
        changed = dict(options, deterministic=True)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            pilot.train(self.definitions, weights, resume=first, **changed)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            pilot.train(self.definitions, {"red": 0.5, "blue_above": 0.5}, resume=first, **options)
        corrupted = copy.deepcopy(first)
        corrupted["report"]["condition_counts"]["red"] = 42
        with self.assertRaisesRegex(ValueError, "exposure counts"):
            pilot.train(self.definitions, weights, resume=corrupted, **options)
        incompatible_runtime = copy.deepcopy(first)
        incompatible_runtime["rng"]["runtime"]["torch"] = "different"
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            pilot.train(self.definitions, weights, resume=incompatible_runtime, **options)
        changed_code = copy.deepcopy(first)
        changed_code["rng"]["runtime"]["code_sha256"]["render"] = "different"
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            pilot.train(self.definitions, weights, resume=changed_code, **options)
        invalid_time = copy.deepcopy(first)
        invalid_time["report"]["elapsed_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "timing or peak-memory"):
            pilot.train(self.definitions, weights, resume=invalid_time, **options)
        invalid_peak = copy.deepcopy(first)
        invalid_peak["report"]["peak_cuda_allocated_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "timing or peak-memory"):
            pilot.train(self.definitions, weights, resume=invalid_peak, **options)

    def test_resume_cli_writes_new_checkpoint_and_protects_input(self):
        torch, pilot = self.torch, self.pilot
        options = dict(steps=1, batch_size=1, seed=9, device="cpu", width=4, timesteps=2,
                       lr=1e-4, dropout=0.1, ema_decay=0.9, max_attempts=10000)
        checkpoint, _ = pilot.train(self.definitions, None, **options)
        with tempfile.TemporaryDirectory() as directory:
            original, output = Path(directory) / "original.pt", Path(directory) / "output.pt"
            pilot.save_checkpoint(original, checkpoint)
            before = original.read_bytes()
            base = [sys.executable, "-m", "exposure.pilot_train", "resume", "--checkpoint", str(original),
                    "--steps", "1"]
            failure = subprocess.run(base + ["--output", str(original)], capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertEqual(original.read_bytes(), before)
            mismatch = subprocess.run(base + ["--output", str(output), "--deterministic"],
                                      capture_output=True, text=True)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("Incompatible", mismatch.stderr)
            self.assertFalse(output.exists())
            run = subprocess.run(base + ["--output", str(output)], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(run.stdout)["config"]["steps"], 2)
            self.assertEqual(torch.load(output, map_location="cpu", weights_only=True)["config"]["steps"], 2)

    def test_deterministic_cuda_requires_early_compatible_workspace(self):
        pilot, torch = self.pilot, self.torch
        try:
            with patch.object(pilot, "_CUBLAS_CONFIGURED_BEFORE_INIT", None):
                with patch.dict(os.environ, {}, clear=True), patch.object(torch.cuda, "is_initialized", return_value=False):
                    pilot._configure_determinism(True, "cuda")
                    self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                    self.assertTrue(torch.are_deterministic_algorithms_enabled())
                    self.assertTrue(torch.backends.cudnn.deterministic)
                    self.assertFalse(torch.backends.cudnn.benchmark)
                    with patch.object(torch.cuda, "is_initialized", return_value=True):
                        pilot._configure_determinism(True, "cuda")
                with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "invalid"}), \
                        patch.object(torch.cuda, "is_initialized", return_value=False):
                    with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
                        pilot._configure_determinism(True, "cuda")
            with patch.object(pilot, "_CUBLAS_CONFIGURED_BEFORE_INIT", None), \
                    patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                    patch.object(torch.cuda, "is_initialized", return_value=True):
                with self.assertRaisesRegex(ValueError, "initialized before deterministic setup"):
                    pilot._configure_determinism(True, "cuda")
        finally:
            pilot._configure_determinism(False, "cpu")


if __name__ == "__main__":
    unittest.main()
