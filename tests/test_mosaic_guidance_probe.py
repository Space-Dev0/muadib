import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional')
class GuidanceTests(unittest.TestCase):
    def test_guidance_interpolates_velocity_before_conversion(self):
        from exposure.mosaic_guidance_probe import GuidedVelocityAsNoise
        class Dummy(torch.nn.Module):
            def forward(self, x, t, tokens, drop=None):
                return torch.ones_like(x) * (2 if drop is None else 1)
        x = torch.full((1, 1, 2, 2), 3.)
        a = torch.tensor([0.25])
        t = torch.tensor([0])
        for scale in (0., 1., 2.):
            got = GuidedVelocityAsNoise(Dummy(), a, scale)(x, t, torch.zeros(1, 1, 4))
            want = (1 + scale) * a.sqrt() + x * (1 - a).sqrt()
            self.assertTrue(torch.allclose(got, want))

    def test_scale_one_matches_existing_sampler(self):
        from exposure import pilot_train
        from exposure.mosaic_pilotunet_probe import sample_image, _token_table
        from exposure.mosaic_guidance_probe import GuidedVelocityAsNoise
        from exposure.identity_eps_finetune import _sample
        torch.set_num_threads(1)
        torch.manual_seed(11)
        model = pilot_train.PilotUNet(4, 8).eval()
        beta, a = pilot_train.schedule(8, torch.device('cpu'))
        tokens = _token_table(torch.device('cpu'))[0][:1]
        old = sample_image(model, tokens, beta, a, seed=7, device='cpu')
        new = _sample(GuidedVelocityAsNoise(model, a, 1.), 'standard', tokens=tokens,
                      beta=beta, alpha_bar=a, seed=7, device=torch.device('cpu'))
        self.assertEqual(old.tobytes(), new.tobytes())
