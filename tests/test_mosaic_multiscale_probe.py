import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional')
class MultiscaleProbeTests(unittest.TestCase):
    def test_multiscale_forward_and_backward(self):
        from exposure.mosaic_multiscale_probe import MultiscaleVelocity
        torch.set_num_threads(1)
        model = MultiscaleVelocity(4)
        image = torch.randn(2, 3, 16, 16)
        output = model(image, torch.tensor([0, 1]), torch.tensor([0.2, 0.8]))
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.isfinite(output).all().item())
        output.square().mean().backward()
        self.assertIsNotNone(model.in_conv.weight.grad)

    def test_sampler_reuses_seeded_noise(self):
        from exposure.mosaic_multiscale_probe import sample
        class ZeroVelocity(torch.nn.Module):
            def forward(self, image, labels, t):
                return torch.zeros_like(image)
        model = ZeroVelocity()
        a = sample(model, 'red', 7, 3, 'cpu').tobytes()
        b = sample(model, 'blue', 7, 3, 'cpu').tobytes()
        self.assertEqual(a, b)
