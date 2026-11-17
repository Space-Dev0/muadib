import unittest
import numpy as np
from PIL import Image
from exposure.checker import DetectionError, audit_rules
from exposure.checker_seg_adapter import classify, objects_from_mask
from exposure.image_diagnostics import renderer_instance_map
from exposure.render import render_scene
from exposure.rules import satisfies

class SegmentationAdapterTests(unittest.TestCase):
    def test_components_on_renderer_annotation_masks(self):
        for seed in range(10):
            scene = render_scene(seed)
            mask = renderer_instance_map(scene.objects) > 0
            actual = objects_from_mask(scene.image, mask)
            self.assertEqual(sorted((x.color, x.shape) for x in actual),
                             sorted((x.color, x.shape) for x in scene.objects))
            for name in ('color_red', 'shape_circle'):
                self.assertEqual(satisfies(actual, audit_rules()[name]), satisfies(scene.objects, audit_rules()[name]))

    def test_abstains_on_missing_or_merged_objects(self):
        image = Image.new('RGB', (64,64), 'white')
        with self.assertRaises(DetectionError):
            objects_from_mask(image, np.zeros((64,64),dtype=bool))
        with self.assertRaises(DetectionError):
            objects_from_mask(image, np.ones((64,64),dtype=bool))

    def test_bad_dimensions(self):
        with self.assertRaises(DetectionError):
            classify(Image.new('RGB',(10,10)), np.ones((64,64)), audit_rules()['color_red'])

if __name__ == '__main__': unittest.main()
