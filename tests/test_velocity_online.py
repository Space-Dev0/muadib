import copy
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is unavailable")
class OnlineVelocityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import pilot_train, velocity_online
        torch.set_num_threads(1)
        cls.torch, cls.pilot, cls.online = torch, pilot_train, velocity_online
        cls.rules = {
            "red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}],
            "blue": [{"type": "attribute", "variable": "a", "field": "color", "value": "blue"}],
        }
        cls.options = dict(steps=2, batch_size=2, seed=17, device="cpu", width=4, timesteps=8,
                           lr=1e-4, dropout=0.0, ema_decay=0.9, max_attempts=10000, deterministic=True)

    def test_exposure_accounting_and_deterministic_replay(self):
        first, report = self.online.train(self.rules, **self.options)
        second, other = self.online.train(self.rules, **self.options)
        self.assertEqual(sum(report["condition_counts"].values()), 4)
        self.assertGreaterEqual(report["sampling_attempts"], 4)
        self.assertEqual(report["condition_counts"], other["condition_counts"])
        self.assertEqual(report["realized_rule_counts_by_condition"], other["realized_rule_counts_by_condition"])
        for condition, count in report["condition_counts"].items():
            self.assertEqual(report["realized_rule_counts_by_condition"][condition][condition], count)
            self.assertTrue(all(0 <= n <= count for n in report["realized_rule_counts_by_condition"][condition].values()))
        for key in first["model"]:
            self.assertTrue(self.torch.equal(first["model"][key], second["model"][key]))
        self.assertTrue(first["diagnostic"]["requires_custom_sampler"])
        self.assertTrue(first["diagnostic"]["resumable"])
        with self.assertRaises(ValueError):
            self.pilot.sample(first, "red", seed=0, device="cpu", guidance=1.0)
        image1 = self.online.sample(first, "red", seed=2, device="cpu")
        image2 = self.online.sample(first, "red", seed=2, device="cpu")
        self.assertEqual(image1.tobytes(), image2.tobytes())

    def test_invalid_input_rejected(self):
        for options in ({"steps": 0}, {"batch_size": True}, {"dropout": -0.1}, {"lr": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.online.train(self.rules, **{**self.options, **options})
        with self.assertRaises(ValueError):
            self.online.train(self.rules, {"red": 1.0}, **self.options)
        with self.assertRaises(ValueError):
            self.online.sample({}, "red", seed=0, device="cpu")

    def test_checkpoint_round_trip(self):
        checkpoint, _ = self.online.train(self.rules, **self.options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            self.pilot.save_checkpoint(path, checkpoint)
            loaded = self.torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(self.online.sample(loaded, "blue", seed=4, device="cpu").size, (64, 64))

    def test_resume_matches_uninterrupted_training_and_preserves_input(self):
        first, _ = self.online.train(self.rules, **self.options)
        prior_model = {k: v.clone() for k, v in first["model"].items()}
        resumed, report = self.online.train(self.rules, **self.options, resume=first)
        uninterrupted, full_report = self.online.train(self.rules, **{**self.options, "steps": 4})
        self.assertEqual(report["condition_counts"], full_report["condition_counts"])
        self.assertEqual(report["realized_rule_counts_by_condition"], full_report["realized_rule_counts_by_condition"])
        self.assertEqual(report["sampling_attempts"], full_report["sampling_attempts"])
        self.assertEqual(resumed["rng"]["python"], uninterrupted["rng"]["python"])
        self.assertTrue(self.torch.equal(resumed["rng"]["torch_cpu"], uninterrupted["rng"]["torch_cpu"]))
        for part in ("model", "ema"):
            for key in resumed[part]:
                self.assertTrue(self.torch.equal(resumed[part][key], uninterrupted[part][key]), key)
        for key in prior_model:
            self.assertTrue(self.torch.equal(first["model"][key], prior_model[key]), key)
        for key, state in resumed["optimizer"]["state"].items():
            for name, value in state.items():
                other = uninterrupted["optimizer"]["state"][key][name]
                self.assertTrue(self.torch.equal(value, other) if self.torch.is_tensor(value) else value == other)

    def test_resume_rejects_older_schema_and_mismatches(self):
        checkpoint, _ = self.online.train(self.rules, **self.options)
        previous = copy.deepcopy(checkpoint)
        previous["schema_version"] = 0
        previous["diagnostic"]["resumable"] = False
        self.assertEqual(self.online.sample(previous, "red", seed=1, device="cpu").size, (64, 64))
        with self.assertRaisesRegex(ValueError, "schema 0"):
            self.online.train(self.rules, **self.options, resume=previous)
        incompatible = copy.deepcopy(checkpoint)
        incompatible["rng"]["runtime"]["code_sha256"]["velocity_online"] = "changed"
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.online.train(self.rules, **self.options, resume=incompatible)
        damaged = copy.deepcopy(checkpoint)
        damaged["report"]["condition_counts"]["red"] += 1
        with self.assertRaisesRegex(ValueError, "exposure"):
            self.online.train(self.rules, **self.options, resume=damaged)

    def test_resume_cli_refuses_to_overwrite_input(self):
        checkpoint, _ = self.online.train(self.rules, **self.options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            self.pilot.save_checkpoint(path, checkpoint)
            before = path.read_bytes()
            result = subprocess.run([sys.executable, "-m", "exposure.velocity_online", "resume",
                                     "--checkpoint", str(path), "--steps", "1", "--output", str(path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
