import importlib.util
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from exposure.mosaic_pilot_data import generate

TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "PyTorch is not installed")
class MosaicToyTrainTests(unittest.TestCase):
    def test_cpu_smoke_matched_initialization_and_finite_losses(self):
        import torch
        from exposure.mosaic_toy_train import train

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            generate(dataset, seed=12, train_per_cell=10, eval_per_complexity=2)
            checkpoints = Path(tmp) / "checkpoints"
            result = train(dataset, seed=13, steps=2, batch_size=2, width=4, device="cpu",
                           checkpoint_dir=checkpoints)
            self.assertEqual(len(result["runs"]), 4)
            self.assertEqual(len({r["initial_parameter_sha256"] for r in result["runs"]}), 1)
            self.assertTrue(all(len(r["losses"]) == 2 and all(math.isfinite(x) for x in r["losses"])
                                for r in result["runs"]))
            self.assertIn("no generated evaluation", result["note"])
            self.assertEqual(len(list(checkpoints.glob("*.pt"))), 4)
            for run in result["runs"]:
                checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["cell"], [run["complexity"], run["imbalance"]])
                self.assertEqual(checkpoint["kind"], "mosaic_toy_velocity")
                self.assertFalse(checkpoint["resumable"])
                self.assertEqual(checkpoint["device"], "cpu")
                self.assertEqual(checkpoint["initial_parameter_sha256"], run["initial_parameter_sha256"])
                self.assertEqual(hashlib.sha256(Path(run["checkpoint"]).read_bytes()).hexdigest(), run["checkpoint_sha256"])
                from exposure.mosaic_toy_sample import sample
                self.assertEqual(sample(checkpoint, label="red", seed=7, steps=2, device="cpu").size, (64, 64))
            with self.assertRaises(FileExistsError):
                train(dataset, seed=13, steps=2, batch_size=2, width=4, device="cpu",
                      checkpoint_dir=checkpoints)

    def test_pixel_tampering_is_rejected(self):
        from PIL import Image
        from exposure.mosaic_toy_train import load_cells

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            result = generate(dataset, seed=12, train_per_cell=10, eval_per_complexity=2)
            Image.new("RGB", (64, 64), "black").save(dataset / result["records"][0]["image"])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_cells(dataset)


if __name__ == "__main__":
    unittest.main()
