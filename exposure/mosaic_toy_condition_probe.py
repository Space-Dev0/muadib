"""Probe paired class-conditioning effects in toy samples, not model accuracy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from exposure import mosaic_toy_sample
from exposure.mosaic_pilot_data import CELLS
from exposure.mosaic_toy_sample import sample


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(checkpoints: Path, *, first_seed: int, seeds: int, steps: int, device: str) -> dict:
    """Use identical initial noise for both labels and all four checkpoints.

    Scores are RGB pixel differences, not a semantic classifier or accuracy.
    """
    if type(first_seed) is not int or first_seed < 0 or type(seeds) is not int or seeds < 1:
        raise ValueError("first_seed must be nonnegative and seeds must be positive integers")
    if type(steps) is not int or steps < 2 or device not in ("cpu", "cuda"):
        raise ValueError("steps must be >=2 and device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if device == "cpu":
        torch.set_num_threads(1)
    common_fields = ("seed", "steps", "batch_size", "width", "learning_rate", "device",
                     "torch_version", "python_version", "manifest_sha256", "trainer_sha256",
                     "initial_parameter_sha256")
    common = None
    results = []
    for complexity, imbalance in CELLS:
        path = checkpoints / f"{complexity}-{imbalance}.pt"
        cp = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(cp, dict) or cp.get("schema_version") != 1 or
                cp.get("kind") != "mosaic_toy_velocity" or cp.get("cell") != [complexity, imbalance]):
            raise ValueError(f"Invalid checkpoint for {complexity}/{imbalance}")
        config = tuple(cp[field] for field in common_fields)
        if common is not None and config != common:
            raise ValueError("Checkpoints do not have matched training settings")
        common = config
        pairs = []
        for seed in range(first_seed, first_seed + seeds):
            images = {}
            for label in ("red", "blue"):
                images[label] = np.asarray(sample(cp, label=label, seed=seed, steps=steps, device=device),
                                           dtype=np.float32) / 255
            delta = images["red"] - images["blue"]
            # Mean(R-B) includes the whole image, including background, and is
            # only a low-level indicator of conditioning—not color accuracy.
            contrast = {label: float(np.mean(x[:, :, 0] - x[:, :, 2])) for label, x in images.items()}
            pairs.append({"seed": seed, "mean_absolute_rgb_difference": float(np.mean(np.abs(delta))),
                          "red_label_mean_red_minus_blue": contrast["red"],
                          "blue_label_mean_red_minus_blue": contrast["blue"],
                          "paired_red_minus_blue_contrast": contrast["red"] - contrast["blue"]})
        results.append({"complexity": complexity, "imbalance": imbalance,
                        "checkpoint_sha256": digest(path), "pairs": pairs,
                        "mean_absolute_rgb_difference": float(np.mean([p["mean_absolute_rgb_difference"] for p in pairs])),
                        "mean_paired_red_minus_blue_contrast": float(np.mean([p["paired_red_minus_blue_contrast"] for p in pairs]))})
    return {"schema_version": 1, "first_seed": first_seed, "seeds": seeds, "steps": steps,
            "device": device, "torch_version": str(torch.__version__),
            "sampler_sha256": digest(Path(mosaic_toy_sample.__file__)),
            "source_sha256": digest(Path(__file__)), "training": dict(zip(common_fields, common, strict=True)),
            "results": results, "note": ("Same initial-noise seeds reused for both labels and all cells; "
            "RGB differences are descriptive only, not semantic validity, image quality, uncertainty, "
            "MOSAIC replication or a Stage 0 gate. The sampler starts at approximate terminal noise.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=81231)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        result = compare(args.checkpoints, first_seed=args.first_seed, seeds=args.seeds,
                         steps=args.steps, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
