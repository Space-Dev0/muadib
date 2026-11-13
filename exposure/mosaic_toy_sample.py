"""Sample a toy velocity checkpoint for plumbing checks, not MOSAIC results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from exposure.mosaic_toy_train import TinyVelocity


def velocity_components(noisy, velocity, alpha):
    """Invert x_t=sqrt(a)x_0+sqrt(1-a)eps and its velocity target."""
    return alpha.sqrt() * noisy - (1 - alpha).sqrt() * velocity, (1 - alpha).sqrt() * noisy + alpha.sqrt() * velocity


@torch.no_grad()
def sample(checkpoint: dict, *, label: str, seed: int, steps: int, device: str) -> Image.Image:
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "mosaic_toy_velocity" or checkpoint.get("schema_version") != 1:
        raise ValueError("Expected a schema-1 toy velocity checkpoint")
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps < 2:
        raise ValueError("seed must be nonnegative and steps at least 2")
    if label not in ("red", "blue") or device not in ("cpu", "cuda"):
        raise ValueError("Unsupported label or device")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    width = checkpoint.get("width")
    if type(width) is not int or width < 1:
        raise ValueError("Invalid checkpoint width")
    model = TinyVelocity(width).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    image = torch.randn((1, 3, 64, 64), generator=generator, device=device)
    labels = torch.tensor([0 if label == "red" else 1], device=device)
    times = torch.linspace(1, 0, steps, device=device)
    for index, t in enumerate(times):
        alpha = 0.99 - 0.98 * t
        velocity = model(image, labels, t.reshape(1))
        clean, noise = velocity_components(image, velocity, alpha)
        if index + 1 == steps:
            image = clean
        else:
            next_alpha = 0.99 - 0.98 * times[index + 1]
            image = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
    if not torch.isfinite(image).all():
        raise ValueError("Non-finite sample")
    pixels = ((image[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.asarray(pixels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", choices=("red", "blue"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        if args.device == "cpu":
            torch.set_num_threads(1)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        image = sample(checkpoint, label=args.label, seed=args.seed, steps=args.steps, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as file:
            image.save(file, format="PNG")
        print(json.dumps({"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                          "output": str(args.output), "label": args.label, "seed": args.seed,
                          "steps": args.steps, "device": args.device,
                          "note": "Approximate Gaussian start at alpha=0.01; feasibility only, not replication."}))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
