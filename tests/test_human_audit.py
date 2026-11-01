import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError

from exposure.human_audit import _write_json, predicate_manifest, predict, prepare, read_reviews, score
from exposure.render import parse_rule, render_scene
from exposure.rules import SceneObject, satisfies


def sample(identifier, relation=False):
    predicates = [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]
    if relation:
        predicates.append({"type": "relation", "subject": "a", "kind": "above", "object": "b"})
    return {"sample_id": identifier, "image_path": f"{identifier}.png", "rule_name": "rel" if relation else "red",
            "predicates": predicates, "k_attr": 1, "k_rel": int(relation), "gold": True, "checker": False}


class HumanAuditTests(unittest.TestCase):
    def test_predicate_questions_are_independent_and_grouped_by_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_scene(42).image.save(root / "a.png")
            render_scene(43).image.save(root / "b.png")
            first, second = sample("a", True), sample("b")
            second["predicates"][0]["variable"] = "z"
            expanded = predicate_manifest({"samples": [second, first]}, root)
            self.assertEqual(expanded, predicate_manifest({"samples": [first, second]}, root))
            questions = expanded["samples"]
            self.assertEqual([q["source_sample_id"] for q in questions], ["a", "a", "b"])
            self.assertEqual([q["predicate_index"] for q in questions], [0, 1, 0])
            self.assertEqual([q["rule_name"] for q in questions],
                             ["attribute:color:red", "relation:above", "attribute:color:red"])
            self.assertEqual(questions[0]["predicates"], questions[2]["predicates"])
            predictions = predict(expanded, root)
            # Independent human judgments are supplied for each question, not inferred from a rule.
            judgments = {q["sample_id"]: i != 1 for i, q in enumerate(questions)}
            result = score(expanded, judgments, predictions)
            self.assertEqual(result["reviewed"], 3)
            self.assertEqual(set(result["by_rule"]), {"attribute:color:red", "relation:above"})
            self.assertEqual(sum(result["by_rule"]["attribute:color:red"][key]
                                 for key in ("tp", "tn", "fp", "fn", "rejected_human_yes",
                                             "rejected_human_no")), 2)

    def test_predicate_cli_rejects_input_and_image_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, manifest, output = root / "a.png", root / "manifest.json", root / "predicates.json"
            render_scene(42).image.save(image)
            manifest.write_text(json.dumps({"samples": [sample("a", True)]}))
            cmd = [sys.executable, "-m", "exposure.human_audit", "predicates",
                   "--manifest", str(manifest), "--dataset-root", str(root)]
            subprocess.run(cmd + ["--output", str(output)], check=True, capture_output=True, text=True)
            self.assertEqual(len(json.loads(output.read_text())["samples"]), 2)
            before = (manifest.read_bytes(), image.read_bytes())
            for target in (manifest, image):
                result = subprocess.run(cmd + ["--output", str(target)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
            self.assertEqual((manifest.read_bytes(), image.read_bytes()), before)

    def test_singleton_truth_does_not_imply_whole_rule_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"placeholder")
            entry = sample("a")
            entry["predicates"].append({"type": "attribute", "variable": "a",
                                        "field": "shape", "value": "square"})
            entry["k_attr"] = 2
            questions = predicate_manifest({"samples": [entry]}, root)["samples"]
            scene = (SceneObject("red", "circle", 10, 10), SceneObject("blue", "square", 30, 30))
            self.assertTrue(all(satisfies(scene, parse_rule(q["predicates"])) for q in questions))
            self.assertFalse(satisfies(scene, parse_rule(entry["predicates"])))

    def test_sampling_is_blinded_reproducible_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [sample(f"a{i}") for i in range(3)] + [sample(f"b{i}", True) for i in range(3)]
            for entry in samples:
                (root / entry["image_path"]).write_bytes(b"image")
            result = prepare({"samples": samples}, root, per_stratum=2, seed=42)
            self.assertEqual(result, prepare({"samples": list(reversed(samples))}, root, per_stratum=2, seed=42))
            self.assertEqual(len(result["samples"]), 4)
            self.assertEqual(sum(entry["k_rel"] for entry in result["samples"]), 2)
            self.assertTrue(all("gold" not in entry and "checker" not in entry for entry in result["samples"]))
            with self.assertRaises(ValueError):
                prepare({"samples": samples}, root, per_stratum=4, seed=42)
            with self.assertRaisesRegex(ValueError, "Duplicate sample ID"):
                prepare({"samples": [samples[0], samples[0]]}, root, per_stratum=1, seed=42)
            bad = dict(samples[0], k_rel=2)
            with self.assertRaisesRegex(ValueError, "complexity"):
                prepare({"samples": [bad]}, root, per_stratum=1, seed=42)

    def test_path_traversal_and_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            root.mkdir()
            (base / "outside.png").write_bytes(b"image")
            escape = sample("a")
            escape["image_path"] = "../outside.png"
            with self.assertRaisesRegex(ValueError, "escapes"):
                prepare({"samples": [escape]}, root, per_stratum=1, seed=1)
            (root / "link.png").symlink_to(base / "outside.png")
            escape["image_path"] = "link.png"
            with self.assertRaisesRegex(ValueError, "escapes"):
                prepare({"samples": [escape]}, root, per_stratum=1, seed=1)

    def test_scoring_confusion_and_rejections(self):
        samples = [sample(str(i)) for i in range(6)]
        judgments = {"0": True, "1": False, "2": False, "3": True, "4": True, "5": False}
        predictions = {"0": True, "1": False, "2": True, "3": False, "4": None, "5": None}
        result = score({"samples": samples}, judgments, predictions)
        overall = result["overall"]
        for key in ("tp", "tn", "fp", "fn", "rejected_human_yes", "rejected_human_no"):
            self.assertEqual(overall[key], 1)
        self.assertEqual(overall["false_reject_rate"], 0.5)
        self.assertEqual(overall["false_accept_rate"], 0.5)
        self.assertEqual(result["by_rule"]["red"], overall)
        with self.assertRaises(ValueError):
            score({"samples": samples}, judgments, {**predictions, "5": 1})
        with self.assertRaises(ValueError):
            score({"samples": samples}, judgments, {key: value for key, value in predictions.items() if key != "5"})

    def test_checker_prediction_uses_pixels_and_rejects_unsupported_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = render_scene(42)
            scene.image.save(root / "a.png")
            Image.new("RGB", (64, 64), "white").save(root / "b.png")
            manifest = {"samples": [sample("a"), sample("b")]}
            result = predict(manifest, root)
            self.assertEqual(result["a"], satisfies(scene.objects, parse_rule(manifest["samples"][0]["predicates"])))
            self.assertIsNone(result["b"])
            self.assertEqual(set(result), {"a", "b"})
            self.assertNotIn("judgment", json.dumps(result))
            (root / "broken.png").write_bytes(b"not an image")
            with self.assertRaises(UnidentifiedImageError):
                predict({"samples": [sample("broken")]}, root)

    def test_predict_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            root.mkdir()
            Image.new("RGB", (64, 64), "white").save(base / "outside.png")
            record = sample("a")
            record["image_path"] = "../outside.png"
            with self.assertRaisesRegex(ValueError, "escapes"):
                predict({"samples": [record]}, root)
            (root / "link.png").symlink_to(base / "outside.png")
            record["image_path"] = "link.png"
            with self.assertRaisesRegex(ValueError, "escapes"):
                predict({"samples": [record]}, root)

    def test_predict_cli_writes_only_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = render_scene(42)
            scene.image.save(root / "a.png")
            manifest, result = root / "manifest.json", root / "predictions.json"
            manifest.write_text(json.dumps({"samples": [sample("a")]}))
            subprocess.run([sys.executable, "-m", "exposure.human_audit", "predict",
                            "--manifest", str(manifest), "--dataset-root", str(root),
                            "--output", str(result)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.read_text()), {"a": satisfies(scene.objects,
                             parse_rule(sample("a")["predicates"]))})
            before = manifest.read_text()
            failed = subprocess.run([sys.executable, "-m", "exposure.human_audit", "predict",
                                     "--manifest", str(manifest), "--dataset-root", str(root),
                                     "--output", str(manifest)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(manifest.read_text(), before)
            failed = subprocess.run([sys.executable, "-m", "exposure.human_audit", "predict",
                                     "--manifest", str(manifest), "--dataset-root", str(root),
                                     "--output", str(root / "a.png")], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)

    def test_zero_denominators_and_malformed_review(self):
        result = score({"samples": [sample("a")]}, {"a": True}, {"a": None})
        self.assertIsNone(result["overall"]["false_reject_rate"])
        self.assertIsNone(result["overall"]["false_accept_rate"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            path.write_text("sample_id,judgment\na,yes\na,no\n")
            with self.assertRaises(ValueError):
                read_reviews(path)
            path.write_text("sample_id,judgment\na,maybe\n")
            with self.assertRaises(ValueError):
                read_reviews(path)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"image")
            manifest, blinded, reviews, predictions, result = [root / name for name in
                ("manifest.json", "blinded.json", "review.csv", "predictions.json", "result.json")]
            manifest.write_text(json.dumps({"samples": [sample("a")]}))
            subprocess.run([sys.executable, "-m", "exposure.human_audit", "prepare", "--manifest", str(manifest),
                            "--dataset-root", str(root), "--per-stratum", "1", "--seed", "1", "--output", str(blinded)],
                           check=True, capture_output=True, text=True)
            with reviews.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["sample_id", "judgment"])
                writer.writerow(["a", "yes"])
            predictions.write_text('{"a": true}')
            subprocess.run([sys.executable, "-m", "exposure.human_audit", "score", "--manifest", str(blinded),
                            "--reviews", str(reviews), "--predictions", str(predictions), "--output", str(result)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.read_text())["overall"]["tp"], 1)

    def test_cli_output_collision_protection_for_all_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, manifest, reviews, predictions = [root / name for name in
                ("a.png", "manifest.json", "reviews.csv", "predictions.json")]
            render_scene(42).image.save(image)
            manifest.write_text(json.dumps({"samples": [sample("a")]}))
            reviews.write_text("sample_id,judgment\na,yes\n")
            predictions.write_text('{"a":true}')
            sources = (image, manifest, reviews, predictions)
            before = {p: p.read_bytes() for p in sources}
            base = [sys.executable, "-m", "exposure.human_audit"]
            commands = {
                "prepare": base + ["prepare", "--manifest", str(manifest), "--dataset-root", str(root),
                                   "--per-stratum", "1", "--seed", "1"],
                "predict": base + ["predict", "--manifest", str(manifest), "--dataset-root", str(root)],
                "score": base + ["score", "--manifest", str(manifest), "--reviews", str(reviews),
                                 "--predictions", str(predictions)],
            }
            cases = [("prepare", "direct", manifest), ("prepare", "symlink", image),
                     ("prepare", "hardlink", manifest), ("predict", "direct", manifest),
                     ("predict", "symlink", image), ("predict", "hardlink", manifest),
                     ("score", "direct", manifest), ("score", "symlink", reviews),
                     ("score", "hardlink", predictions)]
            for index, (command, alias_type, target) in enumerate(cases):
                with self.subTest(command=command, alias=alias_type, target=target.name):
                    output = target if alias_type == "direct" else root / f"alias-{index}"
                    if alias_type == "symlink":
                        output.symlink_to(target)
                    elif alias_type == "hardlink":
                        output.hardlink_to(target)
                    failed = subprocess.run(commands[command] + ["--output", str(output)],
                                            capture_output=True, text=True)
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertIn("Output must differ", failed.stderr)
                    self.assertEqual({p: p.read_bytes() for p in sources}, before)

    def test_failed_output_replacement_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing.json"
            output.write_text("previous result")
            with patch("exposure.human_audit.os.replace", side_effect=OSError("write failure")):
                with self.assertRaises(OSError):
                    _write_json(output, {"new": True})
            self.assertEqual(output.read_text(), "previous result")
            self.assertEqual(list(root.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
