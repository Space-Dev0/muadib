import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from exposure.image_diagnostics import masks_from_instances, radial_power_spectrum, region_mse, renderer_instance_map
from exposure.render import render_scene


class ImageDiagnosticsTests(unittest.TestCase):
    def test_parseval_and_constant_spectrum(self):
        image = np.random.default_rng(9).integers(0, 256, (16, 20, 3), dtype=np.uint8)
        result = radial_power_spectrum(Image.fromarray(image))
        self.assertAlmostEqual(result['total_power'], result['luminance_variance'], places=14)
        self.assertAlmostEqual(sum(x['power_sum'] for x in result['bins']), result['total_power'], places=14)
        uniform = radial_power_spectrum(Image.new('RGB', (16, 20), 'white'))
        self.assertEqual(uniform['total_power'], 0.0)

    def test_region_values_and_empty_gap(self):
        labels = np.zeros((16, 20), dtype=np.uint8)
        labels[4:12, 2:7] = 1
        labels[4:12, 12:17] = 2
        masks = masks_from_instances(labels, boundary_radius=1, gap_radius=4)
        self.assertTrue(all(mask.any() for mask in masks.values()))
        for i, a in enumerate(masks.values()):
            for b in list(masks.values())[i + 1:]:
                self.assertFalse(np.any(a & b))
        reference = Image.new('RGB', (20, 16), 'black')
        modified = np.zeros((16, 20, 3), dtype=np.uint8)
        modified[masks['interior']] = 255
        report = region_mse(reference, Image.fromarray(modified), labels, gap_radius=4)
        self.assertEqual(report['regions']['interior']['rgb_mse'], 1.0)
        self.assertEqual(report['regions']['boundary']['rgb_mse'], 0.0)
        self.assertEqual(report['regions']['inter_object_gap']['rgb_mse'], 0.0)
        one = (labels == 1).astype(np.uint8)
        self.assertIsNone(region_mse(reference, reference, one)['regions']['inter_object_gap']['rgb_mse'])

    def test_reject_bad_maps_and_mismatched_images(self):
        img = Image.new('RGB', (8, 8), 'black')
        with self.assertRaises(ValueError):
            masks_from_instances(np.zeros((8, 8), dtype=int))
        with self.assertRaises(ValueError):
            masks_from_instances(np.ones((8, 8), dtype=float))
        with self.assertRaises(ValueError):
            region_mse(img, Image.new('RGB', (9, 8)), np.ones((8, 8), dtype=int))

    def test_annotated_render_has_three_instances(self):
        scene = render_scene(7)
        labels = renderer_instance_map(scene.objects)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2, 3})
        result = region_mse(scene.image, scene.image, labels)
        self.assertEqual(result['overall_rgb_mse'], 0.0)

    def test_cli_spectrum_and_alignment_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            image = d / 'img.png'
            Image.new('RGB', (10, 10), 'white').save(image)
            output = d / 'out.json'
            base = [sys.executable, '-m', 'exposure.image_diagnostics', '--image', str(image), '--output', str(output)]
            result = subprocess.run(base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNone(json.loads(output.read_text())['paired_error'])
            result = subprocess.run(base + ['--reference', str(image)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('paired MSE requires', result.stderr)


if __name__ == '__main__':
    unittest.main()
