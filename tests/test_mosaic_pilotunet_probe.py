"""Focused tests for the bounded PilotUNet feasibility probe.

Run from the repository root with::

    python -m unittest discover -s tests -v
"""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from exposure import mosaic_pilotunet_probe as probe
from exposure import pilot_train
from exposure.render import _draw
from exposure.rules import SceneObject
from exposure.velocity_probe import velocity_target, velocity_to_noise


class TokenConditioningTests(unittest.TestCase):
    def test_cell_rules_use_pilot_colour_attribute_tokens(self) -> None:
        tokens, names, definitions, label_to_row = probe._token_table(torch.device("cpu"))
        self.assertEqual(names, ["red", "blue"])
        self.assertEqual(label_to_row, {"red": 0, "blue": 1})
        expected = torch.tensor([[[1, 1, 0, 1]], [[1, 1, 0, 2]]], dtype=torch.long)
        self.assertTrue(torch.equal(tokens, expected))
        self.assertEqual(set(definitions), {"red", "blue"})

    def test_conditioning_changes_model_output(self) -> None:
        tokens, _, _, _ = probe._token_table(torch.device("cpu"))
        model = pilot_train.PilotUNet(4, 8).eval()
        image = torch.randn(2, 3, 64, 64)
        time = torch.tensor([1, 2])
        with torch.no_grad():
            red = model(image, time, tokens[torch.tensor([0, 0])])
            blue = model(image, time, tokens[torch.tensor([1, 1])])
        self.assertEqual(red.shape, image.shape)
        self.assertFalse(torch.allclose(red, blue))


class VelocityTargetTests(unittest.TestCase):
    def test_velocity_to_noise_inverts_velocity_target(self) -> None:
        torch.manual_seed(7)
        clean = torch.randn(2, 3, 8, 8)
        noise = torch.randn_like(clean)
        alpha = torch.tensor([0.7, 0.2]).view(2, 1, 1, 1)
        noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
        velocity = velocity_target(clean, noise, alpha)
        recovered = velocity_to_noise(noisy, velocity, alpha)
        self.assertTrue(torch.allclose(recovered, noise, atol=1e-5, rtol=1e-5))


class SamplerTests(unittest.TestCase):
    def test_sampler_is_deterministic_and_paired_by_seed(self) -> None:
        tokens, _, _, _ = probe._token_table(torch.device("cpu"))
        model = pilot_train.PilotUNet(4, 8).eval()
        beta, alpha_bar = pilot_train.schedule(8, torch.device("cpu"))
        red_a = probe.sample_image(model, tokens[0][None], beta, alpha_bar, seed=3, device="cpu")
        red_b = probe.sample_image(model, tokens[0][None], beta, alpha_bar, seed=3, device="cpu")
        blue = probe.sample_image(model, tokens[1][None], beta, alpha_bar, seed=3, device="cpu")
        self.assertEqual(red_a.mode, "RGB")
        self.assertEqual(red_a.size, (64, 64))
        self.assertEqual(red_a.tobytes(), red_b.tobytes())
        # Same seed gives identical noise streams, so only the token differs.
        self.assertNotEqual(red_a.tobytes(), blue.tobytes())

    def test_checkpoint_round_trip_and_sample(self) -> None:
        definitions = probe.cell_rules()
        config = {"width": 4, "timesteps": 8, "deterministic": False, "preview_seeds": [1]}
        model = pilot_train.PilotUNet(4, 8)
        ema = copy.deepcopy(model).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        checkpoint = probe._make_checkpoint(
            config=config, definitions=definitions, label_to_row={"red": 0, "blue": 1},
            model=model, ema=ema, optimizer=optimizer, step=1, loss_mean=1.0,
            elapsed=0.0, provenance={"note": "test"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            pilot_train.save_checkpoint(path, checkpoint)
            loaded = probe.load_checkpoint(path)
            self.assertEqual(loaded["kind"], probe.CHECKPOINT_KIND)
            self.assertTrue(loaded["diagnostic"]["requires_custom_sampler"])
            image = probe.sample(loaded, "red", seed=1, device="cpu")
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (64, 64))


class MetricTests(unittest.TestCase):
    def test_palette_components_separate_clear_objects(self) -> None:
        scene = _draw((SceneObject("red", "circle", 16, 16),
                       SceneObject("green", "square", 48, 48)))
        metrics = probe.image_metrics(scene.image, "red")
        self.assertGreater(metrics["target_fraction"], 0.0)
        self.assertEqual(metrics["target_components"], 1)
        self.assertGreaterEqual(metrics["foreground_components"], 2)
        self.assertLess(metrics["target_fraction"], 0.2)


if __name__ == "__main__":
    unittest.main()
