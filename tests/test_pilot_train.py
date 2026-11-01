import importlib.util
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


if __name__ == "__main__":
    unittest.main()
