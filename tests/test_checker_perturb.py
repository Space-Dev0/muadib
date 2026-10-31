import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from exposure.checker import DetectionError, _detect_palette, audit as clean_audit, audit_rules, detect
from exposure.checker_perturb import CASES, audit
from exposure.render import render_scene
from exposure.rules import satisfies


class CheckerPerturbationTests(unittest.TestCase):
    def test_darkened_clean_scene_and_unknown_foreground(self):
        for seed in range(30):
            scene = render_scene(seed)
            dark = ImageEnhance.Brightness(scene.image).enhance(0.9)
            self.assertEqual(sorted(scene.objects, key=lambda obj: (obj.x, obj.y)),
                             sorted(detect(dark), key=lambda obj: (obj.x, obj.y)))
        ImageDraw.Draw(dark).rectangle((0, 0, 4, 4), fill=(255, 0, 255))
        with self.assertRaises(DetectionError):
            detect(dark)

    def test_threshold_fallback_rejects_disagreement(self):
        image = Image.new("RGB", (64, 64), "white")
        first = render_scene(1).objects
        second = render_scene(2).objects
        with patch("exposure.checker._detect_palette", side_effect=[DetectionError("initial"), first, second]):
            with self.assertRaises(DetectionError):
                detect(image)
        with patch("exposure.checker._detect_palette", side_effect=[DetectionError("initial"), first,
                                                                    DetectionError("ambiguous")]):
            with self.assertRaises(DetectionError):
                detect(image)

    def test_fallback_recovers_some_perturbed_scenes_without_annotation_input(self):
        recovered = {"bright": 0, "blur": 0}
        for seed in range(50):
            scene = render_scene(seed)
            for name, image in (("bright", ImageEnhance.Brightness(scene.image).enhance(1.1)),
                                ("blur", scene.image.filter(ImageFilter.GaussianBlur(0.5)))):
                try:
                    _detect_palette(image)
                except DetectionError:
                    try:
                        objects = detect(image)
                    except DetectionError:
                        continue
                    recovered[name] += 1
                    self.assertEqual(sorted(objects, key=lambda obj: (obj.x, obj.y)),
                                     sorted(scene.objects, key=lambda obj: (obj.x, obj.y)))
        self.assertTrue(all(count > 0 for count in recovered.values()), recovered)

    def test_clean_matches_existing_audit_and_is_reproducible(self):
        result = audit(8, 123)
        self.assertEqual(result, audit(8, 123))
        clean = result["cases"]["clean"]
        reference = clean_audit(8, 123)
        self.assertEqual((clean["total"], clean["evaluated"], clean["rejected"]),
                         (reference["scenes"], reference["evaluated_scenes"], reference["detector_failures"]))
        for name, rule in reference["rules"].items():
            self.assertTrue(all(clean["rules"][name][key] == value for key, value in rule.items()))

    def test_failure_stratification_and_denominators(self):
        with patch("exposure.checker_perturb.detect", side_effect=DetectionError("unsupported")):
            result = audit(1, 42)
        for case in result["cases"].values():
            self.assertEqual((case["total"], case["evaluated"], case["rejected"]), (1, 0, 1))
            for name, predicates in audit_rules().items():
                truth = satisfies(render_scene(42).objects, predicates)
                cell = case["rules"][name]
                self.assertEqual(cell["ground_truth_positive"], int(truth))
                self.assertEqual(cell["ground_truth_negative"], int(not truth))
                self.assertEqual(cell["detector_failure_positive"], int(truth))
                self.assertEqual(cell["detector_failure_negative"], int(not truth))
                self.assertEqual((cell["positive"], cell["negative"], cell["false_accepts"],
                                  cell["false_rejects"]), (0, 0, 0, 0))

    def test_invalid_inputs(self):
        for samples, seed in ((0, 1), (True, 1), (1.0, 1), (1, False), (1, float("nan"))):
            with self.subTest(samples=samples, seed=seed), self.assertRaises(ValueError):
                audit(samples, seed)

    def test_cli_and_total_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"
            subprocess.run([sys.executable, "-m", "exposure.checker_perturb", "--samples", "4",
                            "--seed", "7", "--output", str(path)], check=True, capture_output=True, text=True)
            result = json.loads(path.read_text())
            self.assertEqual(set(result["cases"]), set(CASES))
            for case in result["cases"].values():
                self.assertEqual(case["evaluated"] + case["rejected"], 4)
                for cell in case["rules"].values():
                    self.assertEqual(cell["ground_truth_positive"] + cell["ground_truth_negative"], 4)
                    self.assertEqual(cell["positive"] + cell["detector_failure_positive"],
                                     cell["ground_truth_positive"])
                    self.assertEqual(cell["negative"] + cell["detector_failure_negative"],
                                     cell["ground_truth_negative"])


if __name__ == "__main__":
    unittest.main()
