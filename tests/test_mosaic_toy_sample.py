import importlib.util
import unittest

TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "PyTorch is not installed")
class ToySamplerTests(unittest.TestCase):
    def test_velocity_inversion(self):
        import torch
        from exposure.mosaic_toy_sample import velocity_components

        clean = torch.tensor([[-1.0, 0.2, 0.9]])
        noise = torch.tensor([[0.3, -0.4, 0.7]])
        for value in (0.01, 0.5, 0.99):
            alpha = torch.tensor(value)
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
            recovered_clean, recovered_noise = velocity_components(noisy, velocity, alpha)
            self.assertTrue(torch.allclose(recovered_clean, clean, atol=1e-6))
            self.assertTrue(torch.allclose(recovered_noise, noise, atol=1e-6))
            next_alpha = torch.tensor(0.75)
            moved = next_alpha.sqrt() * recovered_clean + (1 - next_alpha).sqrt() * recovered_noise
            expected = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
            self.assertTrue(torch.allclose(moved, expected, atol=1e-6))

    def test_seeded_cpu_sampling_and_validation(self):
        import torch
        from exposure.mosaic_toy_sample import sample
        from exposure.mosaic_toy_train import TinyVelocity

        torch.set_num_threads(1)
        checkpoint = {"kind": "mosaic_toy_velocity", "schema_version": 1,
                      "width": 4, "state_dict": TinyVelocity(4).state_dict()}
        a = sample(checkpoint, label="red", seed=123, steps=3, device="cpu")
        b = sample(checkpoint, label="red", seed=123, steps=3, device="cpu")
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(a.size, (64, 64))
        with self.assertRaises(ValueError):
            sample(checkpoint, label="red", seed=0, steps=1, device="cpu")
        with self.assertRaises(ValueError):
            sample({}, label="red", seed=0, steps=3, device="cpu")

    def test_oracle_velocity_follows_multistep_path(self):
        import torch
        from exposure.mosaic_toy_sample import velocity_components

        clean = torch.tensor([[-0.8, 0.1, 0.9]])
        noise = torch.tensor([[0.3, -0.4, 0.7]])
        alphas = torch.linspace(0.01, 0.99, 11)
        image = alphas[0].sqrt() * clean + (1 - alphas[0]).sqrt() * noise
        for index, alpha in enumerate(alphas):
            velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
            recovered, predicted_noise = velocity_components(image, velocity, alpha)
            self.assertTrue(torch.allclose(recovered, clean, atol=1e-5))
            if index + 1 < len(alphas):
                next_alpha = alphas[index + 1]
                image = next_alpha.sqrt() * recovered + (1 - next_alpha).sqrt() * predicted_noise
