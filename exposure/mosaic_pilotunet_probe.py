"""Bounded one-cell PilotUNet velocity probe; exploratory, not a Stage 0 result.

The probe trains the repository's ``PilotUNet`` on only the low-complexity,
balanced MOSAIC-inspired cell from a feasibility dataset.  Conditions are the
cell's target colour encoded with the pilot's predicate/rule-token encoder (not
an ad-hoc class embedding).  The network predicts the diffusion velocity and a
DDPM sampler converts velocity to epsilon before each reverse step.  Paired
red/blue previews use the same sampling seed, so their initial and injected
noise are identical and only the conditioning token differs.

This file deliberately does not modify the standard checker, the continuous
training task, existing checkpoints, or existing datasets.  Output directories
must be new, and every checkpoint/report is written exclusively.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from exposure import pilot_train
from exposure import velocity_probe
from exposure.identity_eps_finetune import _sample
from exposure.mosaic_toy_train import load_cells
from exposure.velocity_probe import VelocityAsNoise, velocity_target

CELL = ("low", "balanced")
COLORS = ("red", "blue")
PALETTE = {
    "red": (220, 45, 45),
    "blue": (40, 100, 220),
    "green": (45, 160, 75),
    "yellow": (235, 190, 30),
}
WHITE = np.array((255, 255, 255), dtype=np.float32)
CHECKPOINT_KIND = "mosaic_pilotunet_velocity"
SCHEMA_VERSION = 1


def cell_rules() -> dict:
    """Return the target-colour rules for the low/balanced cell.

    The pilot token encoder expects JSON-style predicates.  Each rule has one
    ``Attribute`` predicate naming the target object's colour.  The subject is
    a single variable, so the encoder produces a one-token condition.
    """
    return {
        "red": [{"type": "attribute", "variable": "target", "field": "color", "value": "red"}],
        "blue": [{"type": "attribute", "variable": "target", "field": "color", "value": "blue"}],
    }


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_device(requested: str) -> str:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access it")
    return requested


def _validate_ints(values: dict, *, minimums: dict) -> None:
    for name, value in values.items():
        if type(value) is not int or value < minimums[name]:
            raise ValueError(f"{name} must be an integer at least {minimums[name]}")


def _token_table(device: torch.device) -> tuple[torch.Tensor, list[str], dict, dict]:
    """Encode the cell rules with the pilot's predicate-token encoder."""
    definitions = cell_rules()
    parsed = pilot_train.load_rules(definitions)
    names = list(parsed)
    if names != list(COLORS):
        raise ValueError("Expected exactly the red and blue target-colour rules")
    tokens = pilot_train.encode_rules(parsed).to(device)
    if tokens.shape != (len(names), 1, 4):
        raise ValueError("Unexpected token shape for the one-predicate cell rules")
    # These assertions pin the semantic mapping: kind 1 is a colour attribute,
    # subject slot 1 is the single target variable, and value 1/2 is red/blue.
    if tokens[:, 0, 0].tolist() != [1, 1] or tokens[:, 0, 1].tolist() != [1, 1]:
        raise ValueError("Unexpected predicate kind or subject slot in cell tokens")
    if tokens[:, 0, 2].tolist() != [0, 0] or tokens[:, 0, 3].tolist() != [1, 2]:
        raise ValueError("Unexpected object slot or colour value in cell tokens")
    return tokens, names, definitions, {name: index for index, name in enumerate(names)}


def load_cell(dataset: Path) -> tuple[torch.Tensor, list[str], str, list[dict], str]:
    """Load and validate only the low/balanced training cell.

    ``load_cells`` already verifies manifest shape, path containment, image
    mode/size and pixel hashes for every training image.  This wrapper selects
    one cell, checks its balanced labels, and returns compact provenance.
    """
    dataset = Path(dataset).resolve(strict=True)
    cells, manifest_hash = load_cells(dataset)
    items = cells[CELL]
    if not items:
        raise ValueError("The low/balanced training cell is empty")
    images = torch.stack([item[0] for item in items])
    colors = ["red" if label == 0 else "blue" for _, label in items]
    if colors.count("red") != colors.count("blue"):
        raise ValueError("The low/balanced training cell is not 50:50")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest.get("records", [])
               if record.get("split") == "train"
               and (record.get("complexity"), record.get("imbalance")) == CELL]
    if len(records) != len(colors):
        raise ValueError("Training records do not match the loaded cell")
    provenance = []
    for record, color in zip(records, colors, strict=True):
        if record.get("target_color") != color:
            raise ValueError("Record order does not match loaded labels")
        provenance.append({"image": record["image"], "target_color": color,
                           "pixel_sha256": record["pixel_sha256"]})
    digest = hashlib.sha256()
    for record in provenance:
        digest.update(record["image"].encode("utf-8"))
        digest.update(record["pixel_sha256"].encode("utf-8"))
    return images, colors, manifest_hash, provenance, digest.hexdigest()


def _runtime(device: str) -> dict:
    """Log the runtime and source fingerprints used by this probe."""
    runtime = pilot_train._runtime(device)
    for name, path in (("mosaic_pilotunet_probe", Path(__file__)),
                       ("mosaic_toy_train", Path(sys.modules["exposure.mosaic_toy_train"].__file__)),
                       ("velocity_probe", Path(velocity_probe.__file__)),
                       ("identity_eps_finetune", Path(sys.modules["exposure.identity_eps_finetune"].__file__))):
        runtime["code_sha256"][name] = _sha256_path(path)
    runtime["cuda_available"] = bool(torch.cuda.is_available())
    runtime["device_used"] = device
    if device == "cuda":
        runtime["cuda_version"] = torch.version.cuda
        runtime["device_name"] = torch.cuda.get_device_name(0)
        runtime["device_capability"] = list(torch.cuda.get_device_capability(0))
    else:
        runtime["device_name"] = "cpu"
        runtime["device_capability"] = None
    return runtime


def _provenance(device: str, dataset: Path, manifest_hash: str, data_hash: str,
                config: dict, definitions: dict, records: list[dict]) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(part) for part in sys.argv],
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "torch": str(torch.__version__),
        "runtime": _runtime(device),
        "dataset": {
            "path": str(Path(dataset).resolve()),
            "manifest_sha256": manifest_hash,
            "selected_cell": list(CELL),
            "training_images": len(records),
            "label_counts": {color: sum(r["target_color"] == color for r in records)
                             for color in COLORS},
            "records_sha256": data_hash,
            "records": records,
        },
        "config": copy.deepcopy(config),
        "rules": copy.deepcopy(definitions),
        "note": ("Provenance for an exploratory one-cell probe. Seeds and hashes aid "
                 "reproduction but do not guarantee bitwise equality across hardware, "
                 "drivers or PyTorch versions."),
    }


def _cpu_state(value):
    """Recursively move optimizer state tensors to CPU for portable checkpoints."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _make_checkpoint(*, config: dict, definitions: dict, label_to_row: dict, model, ema,
                     optimizer, step: int, loss_mean: float, elapsed: float,
                     provenance: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "cell": list(CELL),
        "config": copy.deepcopy(config),
        "rules": copy.deepcopy(definitions),
        "label_to_row": dict(label_to_row),
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "ema": {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()},
        "optimizer": _cpu_state(optimizer.state_dict()),
        "report": {
            "step": step,
            "loss_mean_last_100": loss_mean,
            "elapsed_seconds": elapsed,
            "preview_seeds": config["preview_seeds"],
            "note": "Exploratory one-cell probe; not a Stage 0 gate or model-performance claim.",
        },
        "provenance": copy.deepcopy(provenance),
        "diagnostic": {
            "kind": "mosaic_pilotunet_velocity",
            "prediction": "velocity",
            "sampler": "ddpm_velocity_to_noise",
            "requires_custom_sampler": True,
            "resumable": False,
            "note": "Velocity weights require velocity-to-noise conversion during sampling.",
        },
    }


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("kind") != CHECKPOINT_KIND:
        raise ValueError("Expected a schema-1 mosaic PilotUNet velocity checkpoint")
    diagnostic = checkpoint.get("diagnostic")
    if not isinstance(diagnostic, dict) or diagnostic.get("prediction") != "velocity":
        raise ValueError("Checkpoint does not declare velocity prediction")
    return checkpoint


def _component_count(mask: np.ndarray, min_area: int = 8) -> tuple[int, int]:
    """Count 8-connected components of ``mask`` with at least ``min_area`` pixels."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    areas = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if area >= min_area:
                areas.append(area)
    return len(areas), max(areas, default=0)


def image_metrics(image: Image.Image, target_color: str) -> dict:
    """Return simple palette diagnostics; not a validated quality metric."""
    if target_color not in COLORS:
        raise ValueError("target_color must be red or blue")
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    palette = np.array([PALETTE[color] for color in PALETTE], dtype=np.float32)
    names = list(PALETTE)
    distances = np.linalg.norm(array[:, :, None, :] - palette[None, None, :, :], axis=-1)
    nearest = distances.argmin(axis=-1)
    minimum = distances.min(axis=-1)
    foreground = np.linalg.norm(array - WHITE, axis=-1) > 60
    target_index = names.index(target_color)
    distractor_indices = [names.index("green"), names.index("yellow")]
    target_mask = (nearest == target_index) & (minimum < 80) & foreground
    distractor_mask = np.isin(nearest, distractor_indices) & (minimum < 80) & foreground
    foreground_components, largest_foreground = _component_count(foreground)
    target_components, largest_target = _component_count(target_mask)
    if foreground.any():
        foreground_total = float(foreground.sum())
        max_foreground_fraction = max(
            float(((nearest == index) & foreground).sum()) / foreground_total
            for index in range(len(names)))
    else:
        max_foreground_fraction = 0.0
    return {
        "foreground_fraction": float(foreground.mean()),
        "target_fraction": float(target_mask.mean()),
        "distractor_fraction": float(distractor_mask.mean()),
        "max_foreground_color_fraction": max_foreground_fraction,
        "foreground_components": foreground_components,
        "largest_foreground_component": largest_foreground,
        "target_components": target_components,
        "largest_target_component": largest_target,
        "mean_min_palette_distance": float(minimum.mean()),
    }


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1.0


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode="RGB")


def _save_png(path: Path, image: Image.Image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        image.save(file, format="PNG")
    return _sha256_path(path)


def _sheet(rows: list[list[Image.Image]], path: Path, *, column_labels: list[str],
           row_labels: list[str] | None = None, scale: int = 2, padding: int = 5) -> str:
    """Write a labelled contact sheet without overwriting an existing file."""
    if not rows or any(len(row) != len(column_labels) for row in rows):
        raise ValueError("Sheet rows must match column labels")
    cell = 64 * scale
    header = 14
    left = 112 if row_labels else 0
    width = left + len(column_labels) * cell + (len(column_labels) + 1) * padding
    height = header + len(rows) * cell + (len(rows) + 1) * padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(column_labels):
        x = left + padding + column * (cell + padding)
        draw.text((x, 2), label, fill=(0, 0, 0), font=font)
    for row_index, row in enumerate(rows):
        y = header + padding + row_index * (cell + padding)
        if row_labels:
            draw.text((2, y + cell // 2), row_labels[row_index], fill=(0, 0, 0), font=font)
        for column, image in enumerate(row):
            x = left + padding + column * (cell + padding)
            canvas.paste(image.resize((cell, cell), Image.Resampling.NEAREST), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        canvas.save(file, format="PNG")
    return _sha256_path(path)


def sample_image(model, tokens: torch.Tensor, beta: torch.Tensor, alpha_bar: torch.Tensor,
                 *, seed: int, device: str) -> Image.Image:
    """Sample one condition with the standard DDPM path and velocity conversion."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError("Expected one rule's token tensor with shape (1, predicates, 4)")
    wrapped = VelocityAsNoise(model, alpha_bar)
    return _sample(wrapped, "standard", tokens=tokens, beta=beta, alpha_bar=alpha_bar,
                   seed=seed, device=torch.device(device))


def sample(checkpoint: dict, color: str, *, seed: int, device: str = "auto") -> Image.Image:
    """Load a probe checkpoint and sample one target colour with EMA weights."""
    if checkpoint.get("kind") != CHECKPOINT_KIND or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected a mosaic PilotUNet velocity checkpoint")
    if color not in COLORS:
        raise ValueError("color must be red or blue")
    configuration = checkpoint["config"]
    resolved = _resolve_device(device)
    pilot_train._configure_determinism(bool(configuration.get("deterministic", False)), resolved)
    target = torch.device(resolved)
    parsed = pilot_train.load_rules(checkpoint["rules"])
    names = list(parsed)
    if color not in names:
        raise ValueError("Checkpoint does not contain the requested colour rule")
    tokens = pilot_train.encode_rules(parsed)[names.index(color):names.index(color) + 1].to(target)
    model = pilot_train.PilotUNet(configuration["width"], configuration["timesteps"]).to(target)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    beta, alpha_bar = pilot_train.schedule(configuration["timesteps"], target)
    return sample_image(model, tokens, beta, alpha_bar, seed=seed, device=resolved)


def _preview(output: Path, ema, tokens: torch.Tensor, names: list[str], beta: torch.Tensor,
             alpha_bar: torch.Tensor, *, step: int, seeds: list[int], device: str) -> dict:
    """Generate paired red/blue previews and per-seed contact sheets."""
    model = ema
    records = []
    pairs = []
    images_by_seed = {}
    for seed in seeds:
        pair = {}
        for color in names:
            image = sample_image(model, tokens[names.index(color)][None], beta, alpha_bar,
                                 seed=seed, device=device)
            path = output / f"{step:06d}-{color}-{seed}.png"
            digest = _save_png(path, image)
            record = {"step": step, "seed": seed, "target_color": color, "path": str(path),
                      "sha256": digest, "metrics": image_metrics(image, color)}
            records.append(record)
            pair[color] = (image, record)
        pair_path = output / f"{step:06d}-pair-{seed}.png"
        pair_sha = _sheet([[pair["red"][0], pair["blue"][0]]], pair_path,
                          column_labels=["target=red", "target=blue"],
                          row_labels=[f"seed {seed}"], scale=2)
        pair_record = {"step": step, "seed": seed, "path": str(pair_path), "sha256": pair_sha,
                       "images": [pair["red"][1]["path"], pair["blue"][1]["path"]]}
        pairs.append(pair_record)
        images_by_seed[seed] = [pair["red"][0], pair["blue"][0]]
    grid_path = output / f"{step:06d}-paired-grid.png"
    grid_sha = _sheet([images_by_seed[seed] for seed in seeds], grid_path,
                      column_labels=["target=red", "target=blue"],
                      row_labels=[f"seed {seed}" for seed in seeds], scale=2)
    return {"images": records, "pairs": pairs,
            "grid": {"path": str(grid_path), "sha256": grid_sha,
                     "note": "Each row uses identical sampling noise for red and blue."}}


def _baseline_metrics(baseline_dir: Path | None, step: int, seed: int) -> dict | None:
    if baseline_dir is None:
        return None
    baseline_dir = Path(baseline_dir)
    rows = {}
    for color in COLORS:
        path = baseline_dir / f"{step}-{color}-{seed}.png"
        if not path.is_file():
            return None
        with Image.open(path) as image:
            image.load()
            rows[color] = {"path": str(path), "sha256": _sha256_path(path),
                           "metrics": image_metrics(image, color)}
    return {"path": str(baseline_dir), "step": step, "seed": seed, "images": rows,
            "note": ("Different architecture, sampler and budget from this probe; "
                     "included only as a visual/reference baseline, not a controlled comparison.")}


def _reference_images(images: torch.Tensor, colors: list[str], output: Path) -> dict:
    chosen = {}
    for color in COLORS:
        index = colors.index(color)
        image = _tensor_to_image(images[index])
        path = output / f"reference-{color}.png"
        _save_png(path, image)
        chosen[color] = {"path": str(path), "sha256": _sha256_path(path),
                         "metrics": image_metrics(image, color)}
    return chosen


def _comparison_sheet(output: Path, final_preview: dict, reference: dict, baseline: dict | None) -> str | None:
    if baseline is None:
        return None
    new_by_color = {record["target_color"]: record["path"]
                    for record in final_preview["images"]
                    if record["seed"] == final_preview["images"][0]["seed"]}
    rows = []
    labels = []
    baseline_row = []
    for color in COLORS:
        baseline_path = baseline["images"][color]["path"]
        with Image.open(baseline_path) as image:
            image.load()
            baseline_row.append(image.copy())
    rows.append(baseline_row)
    labels.append(f"baseline {baseline['step']}")
    new_row = []
    for color in COLORS:
        with Image.open(new_by_color[color]) as image:
            image.load()
            new_row.append(image.copy())
    rows.append(new_row)
    labels.append("PilotUNet final")
    reference_row = []
    for color in COLORS:
        with Image.open(reference[color]["path"]) as image:
            image.load()
            reference_row.append(image.copy())
    rows.append(reference_row)
    labels.append("real reference")
    path = output / "comparison-low-balanced.png"
    digest = _sheet(rows, path, column_labels=["target=red", "target=blue"],
                    row_labels=labels, scale=2)
    return {"path": str(path), "sha256": digest,
            "note": "Visual aid only; baseline sampler and budget differ."}


def _snapshot(*, output: Path, step: int, model, ema, optimizer, config: dict,
              definitions: dict, label_to_row: dict, tokens: torch.Tensor, names: list[str],
              beta: torch.Tensor, alpha_bar: torch.Tensor, preview_seeds: list[int],
              device: str, losses: list[float], wall_seconds: float, provenance: dict) -> dict:
    mean = float(np.mean(losses[-100:])) if losses else float("nan")
    checkpoint_path = output / f"pilotunet-{step:06d}.pt"
    checkpoint = _make_checkpoint(config=config, definitions=definitions, label_to_row=label_to_row,
                                  model=model, ema=ema, optimizer=optimizer, step=step, loss_mean=mean,
                                  elapsed=wall_seconds, provenance=provenance)
    pilot_train.save_checkpoint(checkpoint_path, checkpoint)
    previews = _preview(output, ema, tokens, names, beta, alpha_bar,
                        step=step, seeds=preview_seeds, device=device)
    return {
        "step": step,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_path(checkpoint_path),
        "loss_mean_last_100": mean,
        "wall_seconds_since_start": wall_seconds,
        "previews": previews,
    }


def run(dataset: Path, output: Path, *, mode: str, seed: int, steps: int, batch_size: int,
        width: int, timesteps: int, lr: float, dropout: float, ema_decay: float,
        device: str, preview_seeds: list[int], baseline_dir: Path | None = None,
        baseline_step: int = 20000, baseline_seed: int = 81330,
        deterministic: bool = False) -> dict:
    """Train, snapshot and preview the low/balanced cell under a bounded budget."""
    if mode not in ("smoke", "bounded"):
        raise ValueError("mode must be smoke or bounded")
    _validate_ints({"seed": seed, "steps": steps, "batch_size": batch_size, "width": width,
                    "timesteps": timesteps}, minimums={"seed": 0, "steps": 1, "batch_size": 1,
                                                       "width": 4, "timesteps": 2})
    if width % 4:
        raise ValueError("width must be a multiple of four")
    if mode == "smoke" and (steps > 16 or batch_size > 4 or width > 16):
        raise ValueError("Smoke mode is capped at 16 steps, batch 4 and width 16")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("lr must be positive and dropout in [0, 1)")
    if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if not isinstance(preview_seeds, list) or not preview_seeds:
        raise ValueError("At least one preview seed is required")
    if any(type(value) is not int or value < 0 for value in preview_seeds):
        raise ValueError("Preview seeds must be nonnegative integers")
    if len(set(preview_seeds)) != len(preview_seeds):
        raise ValueError("Preview seeds must be unique")
    if type(baseline_step) is not int or baseline_step < 1 or type(baseline_seed) is not int or baseline_seed < 0:
        raise ValueError("Invalid baseline step or seed")
    resolved = _resolve_device(device)
    pilot_train._configure_determinism(bool(deterministic), resolved)
    if resolved == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; select a new directory")
    output.mkdir(parents=True, exist_ok=False)
    if mode == "smoke":
        snapshot_steps = [steps]
    else:
        snapshot_steps = sorted({max(1, steps // 2), steps})
    config = {
        "cell": list(CELL),
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "width": width,
        "timesteps": timesteps,
        "lr": lr,
        "dropout": dropout,
        "ema_decay": ema_decay,
        "deterministic": bool(deterministic),
        "preview_seeds": list(preview_seeds),
        "snapshot_steps": snapshot_steps,
        "sampler": {"name": "ddpm_velocity_to_noise", "steps": timesteps, "guidance": 1.0},
    }
    try:
        target = torch.device(resolved)
        tokens, names, definitions, label_to_row = _token_table(target)
        images, colors, manifest_hash, records, data_hash = load_cell(dataset)
        provenance = _provenance(resolved, dataset, manifest_hash, data_hash, config, definitions, records)
        (output / "provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        images = images.to(target)
        token_indices = torch.tensor([label_to_row[color] for color in colors],
                                     dtype=torch.long, device=target)
        torch.manual_seed(seed)
        model = pilot_train.PilotUNet(width, timesteps).to(target)
        ema = copy.deepcopy(model).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        beta, alpha_bar = pilot_train.schedule(timesteps, target)
        rng = random.Random(seed)
        time_generator = torch.Generator(device=target).manual_seed(seed + 1)
        noise_generator = torch.Generator(device=target).manual_seed(seed + 2)
        drop_generator = torch.Generator(device=target).manual_seed(seed + 3)
        losses: list[float] = []
        loss_trace: list[dict] = []
        snapshots: list[dict] = []
        if resolved == "cuda":
            torch.cuda.synchronize(target)
            torch.cuda.reset_peak_memory_stats(target)
        started = time.perf_counter()
        optimizer_seconds = 0.0
        for step in range(1, steps + 1):
            step_started = time.perf_counter()
            index_list = [rng.randrange(len(colors)) for _ in range(batch_size)]
            indices = torch.tensor(index_list, dtype=torch.long, device=target)
            clean = images[indices]
            t = torch.randint(timesteps, (batch_size,), device=target, generator=time_generator)
            noise = torch.randn(clean.shape, device=target, generator=noise_generator)
            alpha = alpha_bar[t, None, None, None]
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            drop = torch.rand(batch_size, device=target, generator=drop_generator) < dropout
            prediction = model(noisy, t, tokens[token_indices[indices]], drop)
            loss = F.mse_loss(prediction, velocity_target(clean, noise, alpha))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite velocity loss; no report written")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                    average.lerp_(parameter, 1.0 - ema_decay)
            optimizer_seconds += time.perf_counter() - step_started
            value = float(loss.detach().cpu())
            losses.append(value)
            if step == 1 or step % 100 == 0 or step in snapshot_steps:
                loss_trace.append({"step": step, "loss": value})
            if step in snapshot_steps:
                if resolved == "cuda":
                    torch.cuda.synchronize(target)
                wall_seconds = time.perf_counter() - started
                snapshots.append(_snapshot(
                    output=output, step=step, model=model, ema=ema, optimizer=optimizer,
                    config=config, definitions=definitions, label_to_row=label_to_row,
                    tokens=tokens, names=names, beta=beta, alpha_bar=alpha_bar,
                    preview_seeds=preview_seeds, device=resolved, losses=losses,
                    wall_seconds=wall_seconds, provenance=provenance))
        if resolved == "cuda":
            torch.cuda.synchronize(target)
        total_elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(target) if resolved == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved(target) if resolved == "cuda" else None
        reference = _reference_images(images, colors, output)
        baseline = _baseline_metrics(baseline_dir, baseline_step, baseline_seed)
        comparison = _comparison_sheet(output, snapshots[-1]["previews"], reference, baseline)
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "mosaic_pilotunet_velocity_probe",
            "mode": mode,
            "output": str(output),
            "cell": list(CELL),
            "config": config,
            "device": {"requested": device, "resolved": resolved},
            "data": {
                "path": str(Path(dataset).resolve()),
                "manifest_sha256": manifest_hash,
                "records_sha256": data_hash,
                "training_images": len(records),
                "label_counts": {color: colors.count(color) for color in COLORS},
                "records": records,
            },
            "rules": definitions,
            "label_to_row": label_to_row,
            "provenance": provenance,
            "training": {
                "steps_completed": steps,
                "optimizer_seconds": optimizer_seconds,
                "total_elapsed_seconds_including_previews": total_elapsed,
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "loss_mean_last_100": float(np.mean(losses[-100:])),
                "loss_trace": loss_trace,
            },
            "snapshots": snapshots,
            "reference": reference,
            "baseline": baseline,
            "comparison_sheet": comparison,
            "limitations": [
                "Single cell (low/balanced), single training seed and one architecture scale; no architecture ablation.",
                "The dataset is a 200-image synthetic feasibility set, not the MOSAIC benchmark.",
                "No held-out likelihood or object-level validation; previews are qualitative and exploratory.",
                "Metrics are simple palette/connected-component diagnostics, not validated shape or detection scores.",
                "Sampler is 1000-step DDPM with velocity-to-noise conversion; no guidance or sampler-budget sweep.",
                "The baseline comparison uses a different architecture, sampler and budget, so it is illustrative only.",
                "Seeds and provenance do not guarantee bitwise equality across hardware, drivers or PyTorch versions.",
                "This exploratory run is not a Stage 0 gate decision or a claim of MOSAIC replication.",
            ],
            "note": ("Bounded one-cell PilotUNet velocity feasibility probe. It tests whether the stronger "
                     "pilot architecture produces clearer shapes than the small multiscale probe on the same "
                     "low/balanced cell; it does not establish architecture causality or Stage 0 passage."),
        }
        with (output / "report.json").open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        return report
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(),
                   "note": "Partial outputs are preserved; no report.json was written."}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise


def _add_common(parser: argparse.ArgumentParser, *, mode: str) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=81370 if mode == "bounded" else 81371)
    parser.add_argument("--steps", type=int, default=8000 if mode == "bounded" else 3)
    parser.add_argument("--batch-size", type=int, default=8 if mode == "bounded" else 2)
    parser.add_argument("--width", type=int, default=32 if mode == "bounded" else 8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1 if mode == "bounded" else 0.0)
    parser.add_argument("--ema-decay", type=float, default=0.999 if mode == "bounded" else 0.9)
    parser.add_argument("--preview-seeds", type=int, nargs="+",
                        default=[81380, 81381] if mode == "bounded" else [81380])
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--baseline-step", type=int, default=20000)
    parser.add_argument("--baseline-seed", type=int, default=81330)
    parser.add_argument("--deterministic", action="store_true")


def _summary(report: dict) -> dict:
    return {
        "mode": report["mode"],
        "output": report["output"],
        "device": report["device"],
        "config": report["config"],
        "training": report["training"],
        "snapshots": [
            {"step": snapshot["step"], "checkpoint": snapshot["checkpoint"],
             "checkpoint_sha256": snapshot["checkpoint_sha256"],
             "loss_mean_last_100": snapshot["loss_mean_last_100"],
             "paired_grid": snapshot["previews"]["grid"]["path"]}
            for snapshot in report["snapshots"]
        ],
        "comparison_sheet": report["comparison_sheet"],
        "limitations": report["limitations"],
        "note": report["note"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="short end-to-end plumbing and sampler check")
    _add_common(smoke, mode="smoke")
    bounded = commands.add_parser("bounded", help="bounded one-cell training and paired previews")
    _add_common(bounded, mode="bounded")
    generate = commands.add_parser("sample", help="sample a saved velocity checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--color", choices=COLORS, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        if args.command == "sample":
            if args.output.exists():
                raise ValueError("Output already exists; select a new path")
            checkpoint = load_checkpoint(args.checkpoint)
            image = sample(checkpoint, args.color, seed=args.seed, device=args.device)
            digest = _save_png(args.output, image)
            print(json.dumps({"output": str(args.output), "sha256": digest,
                              "color": args.color, "seed": args.seed,
                              "checkpoint_sha256": _sha256_path(args.checkpoint),
                              "note": "Exploratory sample; not a Stage 0 result."}, indent=2))
        else:
            report = run(args.dataset, args.output, mode=args.command, seed=args.seed,
                         steps=args.steps, batch_size=args.batch_size, width=args.width,
                         timesteps=args.timesteps, lr=args.lr, dropout=args.dropout,
                         ema_decay=args.ema_decay, device=args.device,
                         preview_seeds=args.preview_seeds, baseline_dir=args.baseline_dir,
                         baseline_step=args.baseline_step, baseline_seed=args.baseline_seed,
                         deterministic=args.deterministic)
            print(json.dumps(_summary(report), indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
