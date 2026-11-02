"""Exploratory truncated-reconstruction diagnostic for diffusion checkpoints.

Forward-noises an explicit target image to selected timesteps using the actual
checkpoint schedule, then reverses to zero using the existing DDPM epsilon-mean
and posterior variance equations. Compares raw and EMA weights using matched
starting and transition noise on CPU.

This diagnostic is an instrument check and is not proof of sampler correctness,
population quality, or Stage 0 passage.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from exposure.pilot_train import PilotUNet, encode_rules, load_rules, schedule


def load_target(target: Path | str | Image.Image | torch.Tensor) -> tuple[torch.Tensor, Path | None]:
    """Load and normalize a 64x64 target image to [-1, 1] tensor on CPU."""
    if isinstance(target, (str, Path)):
        path = Path(target)
        if not path.is_file():
            raise ValueError(f"Target image file not found: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        target_path = path
    elif isinstance(target, Image.Image):
        image = target.convert("RGB")
        target_path = None
    elif isinstance(target, torch.Tensor):
        t = target.detach().cpu().to(torch.float32)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.ndim != 4 or t.shape != (1, 3, 64, 64):
            raise ValueError(
                f"Target tensor must have shape (1, 3, 64, 64) or (3, 64, 64), got {tuple(target.shape)}"
            )
        if not torch.isfinite(t).all():
            raise ValueError("Target tensor contains non-finite values")
        if (t < -1).any() or (t > 1).any():
            raise ValueError("Target tensor values must be normalized to [-1, 1]")
        return t, None
    else:
        raise TypeError(f"Unsupported target type: {type(target)}")

    if image.size != (64, 64):
        raise ValueError(f"Target image must be 64x64 pixels, got {image.size}")
    pixels = np.array(image, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return tensor, target_path


@torch.no_grad()
def reverse_trajectory(
    model: torch.nn.Module,
    noisy_start: torch.Tensor,
    tokens: torch.Tensor,
    start_timestep: int,
    beta: torch.Tensor,
    alpha_bar: torch.Tensor,
    transition_noises: dict[int, torch.Tensor],
    guidance: float = 1.0,
) -> torch.Tensor:
    """Reverse from start_timestep down to zero using DDPM equations."""
    x = noisy_start.clone()
    for step in range(start_timestep, -1, -1):
        t = torch.tensor([step], dtype=torch.long, device=x.device)
        conditional = model(x, t, tokens)
        if guidance != 1.0:
            unconditional = model(x, t, tokens, torch.ones(1, dtype=torch.bool, device=x.device))
            conditional = unconditional + guidance * (conditional - unconditional)
        x = (x - beta[step] / (1 - alpha_bar[step]).sqrt() * conditional) / (1 - beta[step]).sqrt()
        if step > 0:
            variance = beta[step] * (1 - alpha_bar[step - 1]) / (1 - alpha_bar[step])
            z = transition_noises[step]
            x = x + variance.sqrt() * z
    return x


@torch.no_grad()
def diagnose(
    checkpoint: dict,
    target: Path | str | Image.Image | torch.Tensor,
    *,
    timesteps: list[int],
    seeds: list[int] | int,
    rule: str | None = None,
    guidance: float = 1.0,
    checkpoint_path: Path | str | None = None,
    target_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compare raw and EMA weights across specified timesteps and seeds on CPU."""
    if isinstance(seeds, int):
        seeds = [seeds]
    if (not isinstance(seeds, list) or not seeds or
            any(type(s) is not int or s < 0 for s in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a non-empty list of unique nonnegative integers")

    if (not isinstance(timesteps, list) or not timesteps or
            any(type(t) is not int for t in timesteps) or
            len(set(timesteps)) != len(timesteps)):
        raise ValueError("timesteps must be a non-empty list of unique integers")

    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    for key in ("config", "rules", "model", "ema"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint lacks required key: {key!r}")

    config = checkpoint["config"]
    if not isinstance(config, dict) or "timesteps" not in config or "width" not in config:
        raise ValueError("Checkpoint config must contain 'timesteps' and 'width'")

    total_timesteps = config["timesteps"]
    if type(total_timesteps) is not int or total_timesteps < 2:
        raise ValueError("Checkpoint timesteps must be an integer at least 2")

    for t in timesteps:
        if t < 0 or t >= total_timesteps:
            raise ValueError(f"Timestep {t} is outside schedule [0, {total_timesteps - 1}]")

    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance must be finite and nonnegative")

    clean, derived_path = load_target(target)
    target_display = str(target_path or derived_path) if (target_path or derived_path) else None

    rules = load_rules(checkpoint["rules"])
    if not rules:
        raise ValueError("Checkpoint contains no rules")
    if rule is None:
        if len(rules) == 1:
            rule = next(iter(rules))
        else:
            raise ValueError(f"rule must be specified when checkpoint contains multiple rules: {list(rules)}")
    elif rule not in rules:
        raise ValueError(f"Unknown rule {rule!r}; available rules: {list(rules)}")

    names = list(rules)
    tokens = encode_rules(rules)[names.index(rule):names.index(rule) + 1].to("cpu")

    models = {}
    for name in ("model", "ema"):
        m = PilotUNet(config["width"], total_timesteps).to("cpu")
        m.load_state_dict(checkpoint[name])
        models[name] = m.eval()

    beta, alpha_bar = schedule(total_timesteps, torch.device("cpu"))

    results = []
    for start_step in timesteps:
        abar = alpha_bar[start_step]
        seed_results = []
        for seed in seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            starting_noise = torch.randn(clean.shape, generator=generator, device="cpu")
            noisy_start = abar.sqrt() * clean + (1.0 - abar).sqrt() * starting_noise
            baseline_mse = float((noisy_start - clean).square().mean())

            transition_noises = {
                step: torch.randn(clean.shape, generator=generator, device="cpu")
                for step in range(start_step, 0, -1)
            }

            recon_raw = reverse_trajectory(
                models["model"], noisy_start, tokens, start_step, beta, alpha_bar, transition_noises, guidance=guidance
            )
            raw_mse = float((recon_raw - clean).square().mean())

            recon_ema = reverse_trajectory(
                models["ema"], noisy_start, tokens, start_step, beta, alpha_bar, transition_noises, guidance=guidance
            )
            ema_mse = float((recon_ema - clean).square().mean())

            if not (math.isfinite(baseline_mse) and math.isfinite(raw_mse) and math.isfinite(ema_mse)):
                raise ValueError("Non-finite diagnostic result")

            seed_results.append({
                "seed": seed,
                "baseline_input_mse": baseline_mse,
                "raw": {
                    "reconstruction_mse": raw_mse,
                },
                "ema": {
                    "reconstruction_mse": ema_mse,
                },
                "model": {
                    "reconstruction_mse": raw_mse,
                },
                "raw_reconstruction_mse": raw_mse,
                "ema_reconstruction_mse": ema_mse,
            })

        avg_baseline = sum(s["baseline_input_mse"] for s in seed_results) / len(seed_results)
        avg_raw = sum(s["raw"]["reconstruction_mse"] for s in seed_results) / len(seed_results)
        avg_ema = sum(s["ema"]["reconstruction_mse"] for s in seed_results) / len(seed_results)

        results.append({
            "timestep": start_step,
            "seeds": seed_results,
            "averages": {
                "baseline_input_mse": avg_baseline,
                "raw": {
                    "reconstruction_mse": avg_raw,
                },
                "ema": {
                    "reconstruction_mse": avg_ema,
                },
                "model": {
                    "reconstruction_mse": avg_raw,
                },
                "raw_reconstruction_mse": avg_raw,
                "ema_reconstruction_mse": avg_ema,
            },
        })

    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "target": target_display,
        "rule": rule,
        "schedule_timesteps": total_timesteps,
        "timesteps": timesteps,
        "seeds": seeds,
        "checkpoint_steps": config.get("steps"),
        "results": results,
        "note": ("Truncated reconstruction diagnostic; this separate reverse loop is an "
                 "instrument check and is not proof of sampler correctness, population "
                 "quality or Stage 0 passage."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Truncated reconstruction diagnostic comparing raw and EMA weights"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to pilot or static-scene checkpoint")
    parser.add_argument("--target", type=Path, required=True, help="path to explicit 64x64 target image")
    parser.add_argument("--rule", type=str, default=None, help="rule name (required if checkpoint has multiple rules)")
    parser.add_argument("--timesteps", "--timestep", dest="timesteps", type=int, nargs="+", default=[20, 100, 500],
                        help="start timesteps in [0, schedule-1]")
    parser.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=[0],
                        help="one or more random seeds")
    parser.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance scale (default: 1.0)")
    parser.add_argument("--output", type=Path, default=None, help="optional new JSON output path; never overwrites")
    args = parser.parse_args()

    try:
        if not args.checkpoint.is_file():
            raise ValueError(f"Checkpoint file not found: {args.checkpoint}")
        if not args.target.is_file():
            raise ValueError(f"Target image file not found: {args.target}")

        if args.output is not None:
            if args.output.exists():
                raise ValueError(f"Output path already exists: {args.output}")
            resolved_output = args.output.resolve()
            if resolved_output == args.checkpoint.resolve():
                raise ValueError("Output path cannot overwrite checkpoint")
            if resolved_output == args.target.resolve():
                raise ValueError("Output path cannot overwrite target image")

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        report = diagnose(
            checkpoint,
            args.target,
            timesteps=args.timesteps,
            seeds=args.seeds,
            rule=args.rule,
            guidance=args.guidance,
            checkpoint_path=args.checkpoint,
            target_path=args.target,
        )
        content = json.dumps(report, indent=2, allow_nan=False) + "\n"

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as f:
                f.write(content)
        else:
            print(content, end="")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
