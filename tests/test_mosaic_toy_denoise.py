import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from exposure.mosaic_pilot_data import generate


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class MosaicToyDenoiseTests(unittest.TestCase):
    def test_paired_checkpoints_oracle_and_mismatch(self):
        import torch
        from exposure.mosaic_toy_denoise import diagnose
        from exposure.mosaic_toy_train import train

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, checkpoints = root / "dataset", root / "checkpoints"
            generate(dataset, seed=19, train_per_cell=10, eval_per_complexity=2)
            train(dataset, seed=20, steps=1, batch_size=2, width=4, device="cpu", checkpoint_dir=checkpoints)
            result = diagnose(dataset, checkpoints, seed=21, device="cpu")
            self.assertEqual(result["evaluation_images"], 4)
            self.assertEqual(len(result["results"]), 40)
            for alpha in (0.01, 0.5, 0.99):
                for complexity in ("low", "high"):
                    rows = [r for r in result["results"] if r["alpha"] == alpha and r["evaluation_complexity"] == complexity]
                    self.assertEqual(len(rows), 4)
                    self.assertEqual(len({r["copy_noisy_mse"] for r in rows}), 1)
                    self.assertTrue(all(r["oracle_x0_mse"] < 1e-12 for r in rows))
            path = checkpoints / "high-balanced.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint["steps"] += 1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "matched"):
                diagnose(dataset, checkpoints, seed=21, device="cpu")

    def test_duplicate_training_evaluation_image_is_rejected(self):
        from exposure.mosaic_toy_denoise import _load_images

        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            data = generate(dataset, seed=19, train_per_cell=10, eval_per_complexity=2)
            records = data["records"]
            records[-1]["image"] = records[0]["image"]
            (dataset / "manifest.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "duplicated"):
                _load_images(dataset)
