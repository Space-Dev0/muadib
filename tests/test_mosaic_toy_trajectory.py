"""Tests for the paired trajectory diagnostic."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class TrajectoryTests(unittest.TestCase):
    def test_oracle_recovers_clean_image_at_multiple_step_counts(self):
        import torch
        from exposure.mosaic_toy_trajectory import trajectory
        torch.manual_seed(17)
        clean, noise = torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)
        for steps in (2, 10, 50):
            result = trajectory(clean, noise, lambda x, t, a: a.sqrt() * noise - (1-a).sqrt() * clean, steps)
            self.assertLess(float(torch.mean((result-clean).square())), 1e-11)

    def test_rejects_invalid_inputs(self):
        import torch
        from exposure.mosaic_toy_trajectory import trajectory
        clean = torch.zeros(1, 3, 8, 8)
        with self.assertRaises(ValueError):
            trajectory(clean, clean, lambda x, t, a: x, 1)
        with self.assertRaises(ValueError):
            trajectory(clean, clean[:, :, :2], lambda x, t, a: x, 2)
        with self.assertRaises(ValueError):
            trajectory(clean, clean, lambda x, t, a: x * float("nan"), 2)


if __name__ == "__main__":
    unittest.main()
