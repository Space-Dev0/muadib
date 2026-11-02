import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional dependency is not installed")
class CustomPredictorGuardTests(unittest.TestCase):
    def test_standard_tools_reject_explicit_custom_predictor(self):
        from exposure import high_t_baseline, high_t_finetune, pilot_train, reconstruction_probe

        checkpoint = {"diagnostic": {"requires_custom_sampler": True}}
        operations = (
            lambda: pilot_train.sample(checkpoint, "red", seed=0, device="cpu", guidance=1),
            lambda: high_t_baseline.diagnose(checkpoint, None, timesteps=[0], seeds=[0]),
            lambda: high_t_finetune._checkpoint_parts(checkpoint),
            lambda: reconstruction_probe.diagnose(checkpoint, None, timesteps=[0], seeds=[0]),
        )
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "custom sampler"):
                operation()

    def test_standard_and_unmarked_diagnostics_are_not_rejected(self):
        from exposure.pilot_train import require_standard_predictor

        for checkpoint in ({}, {"diagnostic": {"requires_custom_sampler": False}},
                           {"diagnostic": {"kind": "single_fixed_render"}}):
            require_standard_predictor(checkpoint)
