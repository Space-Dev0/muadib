import importlib.util
import unittest
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ToyOverfitTests(unittest.TestCase):
    def test_repeatable_and_validates_inputs(self):
        from exposure.mosaic_toy_overfit import diagnose

        image = Image.new("RGB", (64, 64), "white")
        first = diagnose(image, label="red", seed=9, steps=2, width=4)
        second = diagnose(image, label="red", seed=9, steps=2, width=4)
        self.assertEqual(first, second)
        self.assertEqual(set(first["after_x0_mse"]), {"0.01", "0.5", "0.99"})
        with self.assertRaises(ValueError):
            diagnose(Image.new("RGB", (16, 16)), label="red", seed=9, steps=2, width=4)
