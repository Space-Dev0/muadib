"""Paired one-step denoising diagnostic for the toy MOSAIC feasibility models.

This measures reconstruction on known held-out images. It does not generate
images from noise and cannot establish a MOSAIC replication or Stage 0 result.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from exposure.measure import unique_object
from exposure.mosaic_pilot_data import CELLS
from exposure.mosaic_toy_sample import velocity_components
from exposure.mosaic_toy_train import TinyVelocity


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_images(root: Path) -> tuple[list[dict], str]:
    """Verify all image identities and return a balanced held-out set."""
    root = root.resolve(strict=True)
    manifest = root / "manifest.json"
    obj = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(obj, dict) or not isinstance(obj.get("records"), list):
        raise ValueError("Expected dataset records")
    hashes = set()
    paths = set()
    evaluation = []
    for record in obj["records"]:
        if not isinstance(record, dict) or record.get("split") not in ("train", "evaluation"):
            raise ValueError("Invalid dataset split")
        path = (root / record["image"]).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path in paths:
            raise ValueError("Dataset image path is duplicated or escapes the root")
        paths.add(path)
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (64, 64):
                raise ValueError("Expected 64x64 RGB images")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest != record.get("pixel_sha256") or digest in hashes:
                raise ValueError("Image hash mismatch or duplicate image")
            hashes.add(digest)
            if record["split"] == "evaluation":
                complexity, label = record.get("complexity"), record.get("target_color")
                if complexity not in ("low", "high") or label not in ("red", "blue"):
                    raise ValueError("Invalid evaluation condition")
                clean = torch.from_numpy(np.array(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 127.5 - 1
                evaluation.append({"name": record["image"], "complexity": complexity,
                                   "label": label, "clean": clean})
    counts = Counter((row["complexity"], row["label"]) for row in evaluation)
    if set(counts) != {(c, label) for c in ("low", "high") for label in ("red", "blue")} or len(set(counts.values())) != 1:
        raise ValueError("Evaluation set must be balanced across complexity and label")
    return evaluation, _sha(manifest)


@torch.no_grad()
def diagnose(dataset: Path, checkpoints: Path, *, seed: int, device: str) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    if device == "cpu":
        torch.set_num_threads(1)
    images, manifest_hash = _load_images(dataset)
    models, checkpoint_hashes, common = {}, {}, None
    match_keys = ("seed", "steps", "batch_size", "width", "learning_rate", "device",
                  "torch_version", "python_version", "manifest_sha256", "trainer_sha256",
                  "initial_parameter_sha256")
    for cell in CELLS:
        path = checkpoints / f"{cell[0]}-{cell[1]}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1 or
                checkpoint.get("kind") != "mosaic_toy_velocity" or checkpoint.get("cell") != list(cell) or
                checkpoint.get("manifest_sha256") != manifest_hash):
            raise ValueError(f"Invalid or mismatched checkpoint for {cell}")
        matching = tuple(checkpoint[key] for key in match_keys)
        if common is None:
            common = matching
        elif common != matching:
            raise ValueError("Checkpoints do not have matched training configuration")
        model = TinyVelocity(checkpoint["width"]).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[cell] = model
        checkpoint_hashes["/".join(cell)] = _sha(path)
    # Noise is generated on CPU once per image, reused across models and alphas.
    rng = torch.Generator(device="cpu").manual_seed(seed)
    grouped = defaultdict(list)
    for row in images:
        clean = row["clean"].unsqueeze(0).to(device)
        noise = torch.randn(clean.shape, generator=rng).to(device)
        label = torch.tensor([0 if row["label"] == "red" else 1], device=device)
        for alpha_value in (0.01, 0.1, 0.5, 0.9, 0.99):
            alpha = torch.tensor(alpha_value, device=device)
            t = torch.tensor([(0.99 - alpha_value) / 0.98], device=device)
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            oracle = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
            recovered_oracle, _ = velocity_components(noisy, oracle, alpha)
            for cell, model in models.items():
                recovered, _ = velocity_components(noisy, model(noisy, label, t), alpha)
                values = {"model_x0_mse": float(F.mse_loss(recovered, clean)),
                          "copy_noisy_mse": float(F.mse_loss(noisy, clean)),
                          "zero_image_mse": float(torch.mean(clean.square())),
                          "oracle_x0_mse": float(F.mse_loss(recovered_oracle, clean))}
                if not all(np.isfinite(value) for value in values.values()):
                    raise ValueError("Non-finite diagnostic measurement")
                grouped[(cell[0], cell[1], row["complexity"], str(alpha_value))].append(values)
    summaries = []
    for (complexity, imbalance, eval_complexity, alpha), values in sorted(grouped.items()):
        summaries.append({"training_complexity": complexity, "imbalance": imbalance,
                          "evaluation_complexity": eval_complexity, "alpha": float(alpha),
                          "image_count": len(values),
                          **{key: float(np.mean([item[key] for item in values])) for key in values[0]}})
    return {"schema_version": 1, "seed": seed, "device": device, "evaluation_images": len(images),
            "manifest_sha256": manifest_hash, "checkpoint_sha256": checkpoint_hashes,
            "source_sha256": _sha(Path(__file__)), "torch_version": str(torch.__version__),
            "training": dict(zip(match_keys, common, strict=True)), "results": summaries,
            "note": ("Paired one-step reconstruction on known images; noise is shared across models. "
                     "Not generated-image accuracy, a MOSAIC replication, or a Stage 0 gate.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=81215)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        result = diagnose(args.dataset, args.checkpoints, seed=args.seed, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
