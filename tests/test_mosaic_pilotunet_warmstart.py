"""Focused tests for the separate PilotUNet warm-start module.

Run from the repository root with::

    python -m unittest discover -s tests -v

The tests use tiny synthetic tensors and a synthetic legacy checkpoint.  They
do not touch the real dataset or any existing checkpoint/output directory.
"""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exposure import mosaic_pilotunet_probe as probe
from exposure import mosaic_pilotunet_warmstart as warm
from exposure import pilot_train


def _legacy_checkpoint(width: int = 4, timesteps: int = 8):
    """Build a schema-1 legacy velocity checkpoint with real optimizer state."""
    torch.manual_seed(11)
    model = pilot_train.PilotUNet(width, timesteps)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.tensor([[[1, 1, 0, 1]], [[1, 1, 0, 2]]], dtype=torch.long)
    image = torch.randn(2, 3, 64, 64)
    time = torch.tensor([1, 2])
    prediction = model(image, time, tokens)
    prediction.square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
            average.lerp_(parameter, 0.1)
    config = {
        "cell": list(probe.CELL),
        "seed": 5,
        "steps": 1,
        "batch_size": 2,
        "width": width,
        "timesteps": timesteps,
        "lr": 1e-3,
        "dropout": 0.0,
        "ema_decay": 0.9,
        "deterministic": False,
        "preview_seeds": [1],
        "snapshot_steps": [1],
        "sampler": {"name": "ddpm_velocity_to_noise", "steps": timesteps, "guidance": 1.0},
    }
    checkpoint = probe._make_checkpoint(
        config=config, definitions=probe.cell_rules(),
        label_to_row={"red": 0, "blue": 1}, model=model, ema=ema,
        optimizer=optimizer, step=1, loss_mean=1.0, elapsed=0.0,
        provenance={"note": "synthetic test checkpoint",
                    "dataset": {"manifest_sha256": "manifest-test",
                                "records_sha256": "records-test"}})
    return checkpoint


def _data_info() -> dict:
    return {"path": "synthetic", "manifest_sha256": "manifest-test",
            "records_sha256": "records-test", "training_images": 4,
            "label_counts": {"red": 2, "blue": 2}, "records": []}


def _train_once(source: dict, output: Path, *, mode: str, seed, steps: int,
                images: torch.Tensor | None = None):
    config = warm._resolve_config(source, mode=mode, seed=seed, steps=steps,
                                  batch_size=None, lr=None, dropout=None,
                                  ema_decay=None, deterministic=None,
                                  preview_seeds=[], snapshot_steps=None)
    source_info = warm._source_info(Path("synthetic-source.pt"), "source-digest",
                                    source, exact=mode == "resume")
    if images is None:
        images = torch.randn(4, 3, 64, 64)
    return warm._train(
        source, source_info=source_info, images=images,
        colors=["red", "blue", "red", "blue"], data_info=_data_info(),
        output=output, config=config, provenance={"note": "test"},
        definitions=probe.cell_rules(), label_to_row={"red": 0, "blue": 1},
        device="cpu")


class RngStateTests(unittest.TestCase):
    def test_capture_and_restore_reproduces_every_stream(self) -> None:
        rng = warm.fresh_rng(21, "cpu")
        state = warm.capture_rng(rng, "cpu", captured_at_step=0)
        expected = (
            [rng.python_random.randrange(1000) for _ in range(3)],
            torch.rand(3, generator=rng.time).tolist(),
            torch.rand(3, generator=rng.noise).tolist(),
            torch.rand(3, generator=rng.drop).tolist(),
            torch.rand(3).tolist(),
            float(np.random.rand()),
        )
        restored = warm.restore_rng(copy.deepcopy(state), "cpu")
        actual = (
            [restored.python_random.randrange(1000) for _ in range(3)],
            torch.rand(3, generator=restored.time).tolist(),
            torch.rand(3, generator=restored.noise).tolist(),
            torch.rand(3, generator=restored.drop).tolist(),
            torch.rand(3).tolist(),
            float(np.random.rand()),
        )
        self.assertEqual(actual, expected)
        self.assertEqual(restored.source, "restored")
        self.assertFalse(restored.not_exact_resume)

    def test_fresh_streams_are_reproducible_and_marked_not_exact(self) -> None:
        first = warm.fresh_rng(33, "cpu")
        second = warm.fresh_rng(33, "cpu")
        self.assertTrue(first.not_exact_resume)
        self.assertEqual(first.source, "fresh_seed")
        self.assertEqual(len(set(first.stream_seeds.values())), 4)
        self.assertEqual(first.python_random.randrange(10_000),
                         second.python_random.randrange(10_000))
        self.assertTrue(torch.equal(torch.rand(4, generator=first.noise),
                                    torch.rand(4, generator=second.noise)))


class RestoreTests(unittest.TestCase):
    def test_warm_start_restores_model_ema_and_optimizer_but_not_rng(self) -> None:
        source = _legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            pilot_train.save_checkpoint(path, source)
            loaded = warm.load_source_checkpoint(path)
            report = _train_once(loaded, Path(directory) / "run", mode="warmstart",
                                 seed=99, steps=2)
            self.assertTrue(report["warm_start"]["not_exact_resume"])
            self.assertFalse(report["warm_start"]["exact_resume"])
            self.assertEqual(report["warm_start"]["rng"]["mode"], "fresh_seed")
            self.assertEqual(report["source_checkpoint"]["restored"],
                             ["model", "ema", "optimizer"])
            checks = report["warm_start"]["restored_state"]
            self.assertTrue(checks["matches_source"])
            self.assertEqual(checks["source_model_sha256"], checks["restored_model_sha256"])
            self.assertEqual(checks["source_optimizer_sha256"],
                             checks["restored_optimizer_sha256"])

            checkpoint = warm.load_warmstart_checkpoint(
                Path(directory) / "run" / "warmstart-000003.pt")
        self.assertEqual(checkpoint["kind"], warm.CHECKPOINT_KIND)
        self.assertTrue(checkpoint["diagnostic"]["resumable"])
        self.assertTrue(checkpoint["optimizer"]["state"])
        self.assertTrue(warm.RNG_KEYS <= checkpoint["rng"].keys())
        self.assertTrue(checkpoint["warm_start"]["not_exact_resume"])

    def test_resume_is_exact_against_an_uninterrupted_run(self) -> None:
        source = _legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = torch.randn(4, 3, 64, 64)
            # Continuous warm start: steps 2, 3 and 4.
            continuous = _train_once(source, root / "continuous", mode="warmstart",
                                     seed=77, steps=3, images=images)
            # Warm start for two steps, then resume for the final step.
            first = _train_once(source, root / "first", mode="warmstart",
                                seed=77, steps=2, images=images)
            final_checkpoint = warm.load_warmstart_checkpoint(
                Path(first["snapshots"][-1]["checkpoint"]))
            resumed = _train_once(final_checkpoint, root / "resumed", mode="resume",
                                  seed=None, steps=1, images=images)
        final_step = continuous["snapshots"][-1]["step"]
        continuous_final = continuous["snapshots"][-1]
        resumed_final = resumed["snapshots"][-1]
        self.assertEqual(final_step, 4)
        self.assertEqual(resumed_final["step"], final_step)
        self.assertFalse(resumed["warm_start"]["not_exact_resume"])
        self.assertTrue(resumed["warm_start"]["exact_resume"])
        self.assertTrue(resumed["warm_start"]["rng"]["matches_checkpoint"])
        self.assertEqual(resumed_final["last_loss"],
                         continuous_final["last_loss"])
        self.assertEqual(resumed_final["model_sha256"], continuous_final["model_sha256"])
        self.assertEqual(resumed_final["ema_sha256"], continuous_final["ema_sha256"])
        self.assertEqual(resumed_final["optimizer_sha256"],
                         continuous_final["optimizer_sha256"])


class CheckpointTests(unittest.TestCase):
    def test_warmstart_loader_rejects_legacy_checkpoint(self) -> None:
        source = _legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            pilot_train.save_checkpoint(path, source)
            with self.assertRaises(ValueError):
                warm.load_warmstart_checkpoint(path)

    def test_snapshot_contains_optimizer_and_rng_state(self) -> None:
        source = _legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            report = _train_once(source, Path(directory) / "run", mode="warmstart",
                                 seed=4, steps=2)
            self.assertEqual([s["step"] for s in report["snapshots"]], [2, 3])
            for snapshot in report["snapshots"]:
                self.assertTrue(snapshot["optimizer_state_included"])
                self.assertTrue(snapshot["rng_state_included"])
                self.assertTrue(Path(snapshot["checkpoint"]).is_file())
                checkpoint = warm.load_warmstart_checkpoint(Path(snapshot["checkpoint"]))
                self.assertEqual(checkpoint["report"]["cumulative_step"], snapshot["step"])
                self.assertEqual(checkpoint["diagnostic"]["optimizer_state"], "included")
                self.assertEqual(checkpoint["diagnostic"]["rng_state"], "included")
                self.assertTrue(checkpoint["rng"]["generators"])

    def test_resume_rejects_changed_config_and_data_mismatch(self) -> None:
        source = _legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            first = _train_once(source, Path(directory) / "first", mode="warmstart",
                                seed=1, steps=1)
            checkpoint = warm.load_warmstart_checkpoint(
                Path(first["snapshots"][-1]["checkpoint"]))
        with self.assertRaises(ValueError):
            warm._resolve_config(checkpoint, mode="resume", seed=None, steps=1,
                                 batch_size=3, lr=None, dropout=None, ema_decay=None,
                                 deterministic=None, preview_seeds=[], snapshot_steps=None)
        wrong = _data_info()
        wrong["records_sha256"] = "different"
        with self.assertRaises(ValueError):
            warm._validate_data_against_source(checkpoint, wrong)

    def test_sample_from_warmstart_checkpoint(self) -> None:
        source = _legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            report = _train_once(source, Path(directory) / "run", mode="warmstart",
                                 seed=8, steps=1)
            checkpoint = warm.load_warmstart_checkpoint(
                Path(report["snapshots"][-1]["checkpoint"]))
            red = warm.sample(checkpoint, "red", seed=3, device="cpu")
            again = warm.sample(checkpoint, "red", seed=3, device="cpu")
        self.assertIsInstance(red, Image.Image)
        self.assertEqual(red.mode, "RGB")
        self.assertEqual(red.size, (64, 64))
        self.assertEqual(red.tobytes(), again.tobytes())


if __name__ == "__main__":
    unittest.main()
