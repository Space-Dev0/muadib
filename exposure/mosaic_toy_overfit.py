"""Fixed-image toy velocity diagnostic; not a model or Stage 0 evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from exposure.mosaic_toy_sample import velocity_components
from exposure.mosaic_toy_train import TinyVelocity


def diagnose(image: Image.Image, *, label: str, seed: int, steps: int, width: int) -> dict:
    if image.mode != "RGB" or image.size != (64, 64):
        raise ValueError("Expected 64x64 RGB image")
    if label not in ("red", "blue") or type(seed) is not int or seed < 0 or type(steps) is not int or steps < 1 or type(width) is not int or width < 1:
        raise ValueError("Invalid label, seed, steps or width")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    clean = torch.from_numpy(np.array(image, dtype=np.uint8).copy()).permute(2, 0, 1).float()[None] / 127.5 - 1
    labels = torch.tensor([0 if label == "red" else 1])
    model = TinyVelocity(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # Evaluation uses fixed draws generated independently of the training stream.
    generator = torch.Generator().manual_seed(seed + 1)
    eval_noise = torch.randn(clean.shape, generator=generator)

    @torch.no_grad()
    def evaluate():
        result = {}
        for alpha_value in (0.01, 0.5, 0.99):
            alpha = torch.tensor(alpha_value)
            t = torch.tensor([(0.99 - alpha_value) / 0.98])
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * eval_noise
            velocity = model(noisy, labels, t)
            recovered, _ = velocity_components(noisy, velocity, alpha)
            result[str(alpha_value)] = float(F.mse_loss(recovered, clean))
        return result

    before = evaluate()
    for _ in range(steps):
        t = torch.rand(1)
        alpha = (0.99 - 0.98 * t)[:, None, None, None]
        noise = torch.randn_like(clean)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
        loss = F.mse_loss(model(noisy, labels, t), target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite velocity loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"schema_version": 1, "seed": seed, "steps": steps, "width": width,
            "label": label, "before_x0_mse": before, "after_x0_mse": evaluate(),
            "note": "One image and one evaluation noise draw; no full samples or Stage 0 conclusion."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", choices=("red", "blue"), required=True)
    parser.add_argument("--seed", type=int, default=81209)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with Image.open(args.image) as image:
            image.load()
            result = diagnose(image, label=args.label, seed=args.seed, steps=args.steps, width=args.width)
        result["image_sha256"] = hashlib.sha256(args.image.read_bytes()).hexdigest()
        result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["torch_version"] = str(torch.__version__)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
