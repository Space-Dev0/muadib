import unittest

from exposure.render import COLORS, IMAGE_SIZE, RADIUS, SHAPES, render_scene
from exposure.rules import Attribute, Relation, satisfies


class RendererTests(unittest.TestCase):
    def test_seed_reproduces_image_and_annotations(self):
        first, second = render_scene(41), render_scene(41)
        self.assertEqual(first.objects, second.objects)
        self.assertEqual(first.image.tobytes(), second.image.tobytes())
        self.assertEqual(first.image.size, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(first.image.mode, "RGB")

    def test_objects_are_visible_separated_and_match_rules(self):
        for seed in range(50):
            scene = render_scene(seed)
            self.assertEqual(len(scene.objects), 3)
            for i, obj in enumerate(scene.objects):
                self.assertIn(obj.color, COLORS)
                self.assertIn(obj.shape, SHAPES)
                self.assertTrue(RADIUS <= obj.x < IMAGE_SIZE - RADIUS)
                self.assertTrue(RADIUS <= obj.y < IMAGE_SIZE - RADIUS)
                self.assertTrue(satisfies(scene.objects, (Attribute("a", "color", obj.color),)))
                for other in scene.objects[i + 1:]:
                    self.assertTrue(abs(obj.x - other.x) >= 2 * RADIUS + 2 or
                                    abs(obj.y - other.y) >= 2 * RADIUS + 2)
            self.assertTrue(satisfies(scene.objects, (Relation("a", "left_of", "b"),))
                            if len({obj.x for obj in scene.objects}) > 1 else
                            not satisfies(scene.objects, (Relation("a", "left_of", "b"),)))


if __name__ == "__main__":
    unittest.main()
