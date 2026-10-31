import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from exposure.checker import DetectionError, audit, audit_rules, detect
from exposure.render import COLORS, SHAPES, _draw, render_scene
from exposure.rules import Attribute, Relation, SceneObject, satisfies


class CheckerTests(unittest.TestCase):
    def test_clean_images_and_vocabulary(self):
        colors, shapes = set(), set()
        for seed in range(100):
            scene = render_scene(seed)
            objects = detect(scene.image)
            self.assertEqual(sorted(scene.objects, key=lambda o: (o.x, o.y)),
                             sorted(objects, key=lambda o: (o.x, o.y)))
            colors.update(o.color for o in objects)
            shapes.update(o.shape for o in objects)
        self.assertEqual(colors, set(COLORS))
        self.assertEqual(shapes, set(SHAPES))
        self.assertEqual(detect(render_scene(7).image), detect(render_scene(7).image))

    def test_spatial_ties_do_not_become_relations(self):
        scene = (SceneObject("red", "triangle", 12, 12),
                 SceneObject("blue", "square", 32, 12),
                 SceneObject("green", "circle", 52, 52))
        rule = (Attribute("a", "color", "red"), Attribute("b", "color", "blue"),
                Relation("a", "above", "b"))
        self.assertFalse(satisfies(detect(_draw(scene).image), rule))

    def test_rejects_invalid_images(self):
        for image in (Image.new("RGB", (64, 64), "black"), Image.new("RGB", (64, 64), "white"),
                      Image.new("RGB", (20, 20), "white"), Image.new("L", (64, 64), 255)):
            with self.subTest(image=image), self.assertRaises(DetectionError):
                detect(image)

    def test_audit_and_cli(self):
        result = audit(300, 0)
        self.assertEqual(result["detector_failures"], 0)
        for cell in result["rules"].values():
            self.assertEqual(cell["positive"] + cell["negative"], 300)
            self.assertGreater(cell["positive"], 0)
            self.assertGreater(cell["negative"], 0)
            self.assertEqual((cell["false_accepts"], cell["false_rejects"]), (0, 0))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "audit.json"
            subprocess.run([sys.executable, "-m", "exposure.checker", "--samples", "3", "--output", str(output)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text())["scenes"], 3)

    def test_detector_failure_stratified_by_ground_truth(self):
        image = render_scene(0)
        with patch("exposure.checker.detect", side_effect=DetectionError("uncertain")):
            result = audit(1, 0)
        self.assertEqual((result["detector_failures"], result["evaluated_scenes"]), (1, 0))
        for name, predicates in audit_rules().items():
            truth = satisfies(image.objects, predicates)
            cell = result["rules"][name]
            self.assertEqual(cell["detector_failure_positive"], int(truth))
            self.assertEqual(cell["detector_failure_negative"], int(not truth))
            self.assertEqual((cell["false_accepts"], cell["false_rejects"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
