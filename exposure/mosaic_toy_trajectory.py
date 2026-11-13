"""Paired sampler trajectory diagnostic on held-out toy images, not a replication."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from exposure.mosaic_pilot_data import CELLS
from exposure.mosaic_toy_denoise import _load_images, _sha
from exposure.mosaic_toy_sample import velocity_components
from exposure.mosaic_toy_train import TinyVelocity


@torch.no_grad()
def trajectory(clean, noise, predict, steps):
    """Follow the sampler from the exact forward noising of a known image."""
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be an integer of at least 2")
    if clean.shape != noise.shape:
        raise ValueError("clean and noise shapes differ")
    times = torch.linspace(1, 0, steps, device=clean.device)
    alpha = 0.99 - 0.98 * times[0]
    image = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    for index, t in enumerate(times):
        alpha = 0.99 - 0.98 * t
        velocity = predict(image, t, alpha)
        recovered, predicted_noise = velocity_components(image, velocity, alpha)
        if index + 1 == steps:
            image = recovered
        else:
            next_alpha = 0.99 - 0.98 * times[index + 1]
            image = next_alpha.sqrt() * recovered + (1 - next_alpha).sqrt() * predicted_noise
    if not torch.isfinite(image).all():
        raise ValueError("Non-finite trajectory")
    return image


@torch.no_grad()
def diagnose(dataset, checkpoints, *, seed, steps, device):
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps < 2:
        raise ValueError("seed must be nonnegative and steps at least 2")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    if device == "cpu":
        torch.set_num_threads(1)
    images, manifest_hash = _load_images(dataset)
    keys = ("seed", "steps", "batch_size", "width", "learning_rate", "device",
            "torch_version", "python_version", "manifest_sha256", "trainer_sha256",
            "initial_parameter_sha256")
    models, hashes, common = {}, {}, None
    for cell in CELLS:
        path = checkpoints / f"{cell[0]}-{cell[1]}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1 or
            checkpoint.get("kind") != "mosaic_toy_velocity" or checkpoint.get("cell") != list(cell) or
            checkpoint.get("manifest_sha256") != manifest_hash):
            raise ValueError(f"Invalid or mismatched checkpoint for {cell}")
        configuration = tuple(checkpoint[key] for key in keys)
        if common is not None and common != configuration:
            raise ValueError("Checkpoints do not have matched training configuration")
        common = configuration
        model = TinyVelocity(checkpoint["width"]).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[cell] = model
        hashes["/".join(cell)] = _sha(path)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    grouped = defaultdict(list)
    for row in images:
        clean = row["clean"].unsqueeze(0).to(device)
        noise = torch.randn(clean.shape, generator=rng).to(device)
        label = torch.tensor([0 if row["label"] == "red" else 1], device=device)
        oracle = trajectory(clean, noise, lambda x, t, a: a.sqrt() * noise - (1-a).sqrt() * clean, steps)
        oracle_mse = float(F.mse_loss(oracle, clean))
        for cell, model in models.items():
            recovered = trajectory(clean, noise, lambda x, t, a: model(x, label, t.reshape(1)), steps)
            mse = float(F.mse_loss(recovered, clean))
            grouped[(cell[0], cell[1], row["complexity"], row["label"])].append((mse, oracle_mse))
    results = []
    for (complexity, imbalance, eval_complexity, label), values in sorted(grouped.items()):
        results.append({"training_complexity": complexity, "imbalance": imbalance,
                        "evaluation_complexity": eval_complexity, "label": label,
                        "image_count": len(values), "model_mse": float(np.mean([x[0] for x in values])),
                        "oracle_mse": float(np.mean([x[1] for x in values]))})
    return {"schema_version": 1, "seed": seed, "steps": steps, "device": device,
            "evaluation_images": len(images), "manifest_sha256": manifest_hash,
            "checkpoint_sha256": hashes, "source_sha256": _sha(Path(__file__)),
            "torch_version": str(torch.__version__), "training": dict(zip(keys, common, strict=True)),
            "results": results, "note": ("Paired reconstruction from exact forward-noised held-out images; "
            "noise shared across models. Not generated-image accuracy, MOSAIC replication, or a Stage 0 gate.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=81217)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        result = diagnose(args.dataset, args.checkpoints, seed=args.seed, steps=args.steps, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
