import importlib.util
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
        self.assertFalse(first["diagnostic"]["resumable"])
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


if __name__ == "__main__":
    unittest.main()
