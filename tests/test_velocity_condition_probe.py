import copy
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency unavailable")
class VelocityConditionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import velocity_condition_probe, velocity_online
        torch.set_num_threads(1)
        cls.probe = velocity_condition_probe
        cls.rules = {
            "red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}],
            "blue": [{"type": "attribute", "variable": "a", "field": "color", "value": "blue"}],
        }
        cls.checkpoint, _ = velocity_online.train(
            cls.rules, steps=1, batch_size=1, seed=2, device="cpu", width=4, timesteps=8,
            lr=1e-4, dropout=0.1, ema_decay=0.9, max_attempts=10000, deterministic=True)

    def test_paired_losses_and_false_rules_are_reproducible(self):
        from exposure import pilot_train
        from exposure.render import render_conditioned
        from exposure.rules import satisfies
        options = {"scene_seeds": [400, 401, 402], "noise_seed": 98, "timesteps": [1, 6]}
        first = self.probe.diagnose(self.checkpoint, **options)
        self.assertEqual(first, self.probe.diagnose(self.checkpoint, **options))
        rules = pilot_train.load_rules(self.rules)
        self.assertGreater(len(first["rows"]), 0)
        for row in first["rows"]:
            scene, _ = render_conditioned(row["scene_seed"], rules[row["rule"]])
            self.assertTrue(satisfies(scene.objects, rules[row["rule"]]))
            self.assertFalse(satisfies(scene.objects, rules[row["false_rule"]]))
            self.assertAlmostEqual(row["false_minus_correct"], row["loss"]["false"] - row["loss"]["correct"])

    def test_rejects_invalid_schedules_and_checkpoint(self):
        options = {"scene_seeds": [5], "noise_seed": 7, "timesteps": [1]}
        with self.assertRaises(ValueError):
            self.probe.diagnose(self.checkpoint, **{**options, "scene_seeds": [5, 5]})
        with self.assertRaises(ValueError):
            self.probe.diagnose(self.checkpoint, **{**options, "timesteps": [8]})
        invalid = copy.deepcopy(self.checkpoint)
        invalid["diagnostic"]["prediction"] = "epsilon"
        with self.assertRaises(ValueError):
            self.probe.diagnose(invalid, **options)
