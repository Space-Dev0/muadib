import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional pilot dependency is not installed")
class StaticSceneProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from exposure import pilot_train, static_scene_probe
        cls.torch, cls.pilot, cls.probe = torch, pilot_train, static_scene_probe
        torch.set_num_threads(1)
        cls.definitions = {"red": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}]}

    def test_fixed_scene_is_labeled_and_cannot_resume_as_online(self):
        original = self.pilot.render_conditioned
        target, checkpoint, report = self.probe.overfit(
            self.definitions, rule="red", scene_seed=77, train_seed=3, steps=2,
            batch_size=1, width=4, timesteps=2, device="cpu", deterministic=True, max_attempts=10000)
        self.assertIs(self.pilot.render_conditioned, original)
        self.assertEqual(checkpoint["schema_version"], 0)
        self.assertEqual(checkpoint["diagnostic"]["kind"], "single_fixed_render")
        self.assertFalse(checkpoint["diagnostic"]["resumable"])
        self.assertEqual(report["condition_counts"]["red"], 2)
        self.assertTrue(self.pilot.satisfies(target.objects, self.pilot.load_rules(self.definitions)["red"]))
        with self.assertRaisesRegex(ValueError, "version 3"):
            self.pilot._resume_parts(checkpoint)

    def test_renderer_restored_when_training_fails(self):
        original = self.pilot.render_conditioned
        with self.assertRaises(ValueError):
            self.probe.overfit(self.definitions, rule="red", scene_seed=77, train_seed=3, steps=0,
                               batch_size=1, width=4, timesteps=2, device="cpu", deterministic=False,
                               max_attempts=10000)
        self.assertIs(self.pilot.render_conditioned, original)

    def test_cli_outputs_are_new_and_explicitly_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rules, output = root / "rules.json", root / "probe"
            rules.write_text(json.dumps(self.definitions), encoding="utf-8")
            command = [sys.executable, "-m", "exposure.static_scene_probe", "--rules", str(rules),
                       "--rule", "red", "--output-dir", str(output), "--steps", "1", "--batch-size", "1",
                       "--width", "4", "--timesteps", "2", "--device", "cpu"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual({p.name for p in output.iterdir()}, {"target.png", "sample.png", "checkpoint.pt", "report.json"})
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["diagnostic"]["kind"], "single_fixed_render")
            before = (output / "checkpoint.pt").read_bytes()
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual((output / "checkpoint.pt").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
