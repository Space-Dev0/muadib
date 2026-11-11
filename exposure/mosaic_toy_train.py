"""Train four tiny velocity models on the feasibility dataset; not a replication.

This bounded trainer checks matched training plumbing. It does not sample or
evaluate a diffusion model, validate a checker, or establish a research gate.
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from exposure.mosaic_pilot_data import CELLS


class TinyVelocity(nn.Module):
    """Small fully convolutional model with class and scalar time conditioning."""

    def __init__(self, width: int):
        super().__init__()
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.condition = nn.Embedding(2, width)
        self.time = nn.Linear(1, width)
        self.middle = nn.Conv2d(width, width, 3, padding=1)
        self.output = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, noisy, labels, timestep):
        feature = self.input(noisy)
        condition = self.condition(labels) + self.time(timestep[:, None])
        feature = F.silu(feature + condition[:, :, None, None])
        return self.output(F.silu(self.middle(feature)))


def load_cells(root: Path) -> tuple[dict, str]:
    """Load and validate training images; reject path escapes and tampering."""
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("records"), list):
        raise ValueError("Expected a feasibility dataset manifest")
    cells = {cell: [] for cell in CELLS}
    seen = set()
    for record in manifest["records"]:
        if not isinstance(record, dict):
            raise ValueError("Invalid dataset record")
        if record.get("split") != "train":
            continue
        cell = (record.get("complexity"), record.get("imbalance"))
        label = record.get("target_color")
        if cell not in cells or label not in ("red", "blue"):
            raise ValueError("Invalid training cell or label")
        path = (root / record["image"]).resolve(strict=True)
        if not path.is_relative_to(root) or path in seen or not path.is_file():
            raise ValueError("Training image must be unique and inside the dataset")
        seen.add(path)
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (64, 64):
                raise ValueError("Training image must be 64x64 RGB")
            if hashlib.sha256(image.tobytes()).hexdigest() != record.get("pixel_sha256"):
                raise ValueError("Training image pixel hash mismatch")
            tensor = torch.from_numpy(np.array(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 127.5 - 1
        cells[cell].append((tensor, 0 if label == "red" else 1))
    counts = {len(items) for items in cells.values()}
    if len(counts) != 1 or next(iter(counts)) == 0:
        raise ValueError("All four cells must have an equal positive number of images")
    n = next(iter(counts))
    if n % 10:
        raise ValueError("Training cell size must be divisible by ten")
    for (_, imbalance), items in cells.items():
        expected = n // 2 if imbalance == "balanced" else 9 * n // 10
        if sum(label == 0 for _, label in items) != expected:
            raise ValueError("Training labels do not match required 50:50 or 90:10 quotas")
    return cells, hashlib.sha256(raw).hexdigest()


def train(root: Path, *, seed: int, steps: int, batch_size: int, width: int, device: str) -> dict:
    """Train four independent models under the same initialization and budget."""
    if any(type(x) is not int for x in (seed, steps, batch_size, width)) or seed < 0 or min(steps, batch_size, width) < 1:
        raise ValueError("seed must be nonnegative; steps, batch size and width must be positive integers")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    cells, manifest_hash = load_cells(root)
    results = []
    for cell in CELLS:
        # Reset streams so the four cells have matched initialization and
        # random draw schedules, while remaining distinct model instances.
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = random.Random(seed)
        model = TinyVelocity(width).to(device)
        initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        losses = []
        start = time.perf_counter()
        for _ in range(steps):
            examples = [cells[cell][rng.randrange(len(cells[cell]))] for _ in range(batch_size)]
            clean = torch.stack([item[0] for item in examples]).to(device)
            labels = torch.tensor([item[1] for item in examples], device=device)
            t = torch.rand(batch_size, device=device)
            alpha = (0.99 - 0.98 * t)[:, None, None, None]
            noise = torch.randn_like(clean)
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            target = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
            loss = F.mse_loss(model(noisy, labels, t), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in cell {cell}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if device == "cuda":
            torch.cuda.synchronize()
        results.append({"complexity": cell[0], "imbalance": cell[1], "training_images": len(cells[cell]),
                        "initial_parameter_sha256": initial_hash, "losses": losses,
                        "elapsed_seconds": time.perf_counter() - start})
    return {"schema_version": 1, "seed": seed, "steps": steps, "batch_size": batch_size,
            "width": width, "device": device, "learning_rate": 0.0001,
            "manifest_sha256": manifest_hash, "torch_version": str(torch.__version__),
            "python_version": sys.version.split()[0],
            "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runs": results,
            "note": "Feasibility smoke only: no generated evaluation, replication or Stage 0 conclusion."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=81204)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        if args.device == "cpu":
            torch.set_num_threads(1)
        result = train(args.dataset, seed=args.seed, steps=args.steps, batch_size=args.batch_size,
                       width=args.width, device=args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Wrote feasibility report to {args.output}")


if __name__ == "__main__":
    main()
