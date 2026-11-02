"""One-step high-noise baseline audit for pilot checkpoints on CPU.

For a fixed target and matched noise, this module compares the raw and EMA
noise predictions with trivial copy-input, zero-noise, and exact-oracle
predictors. It reports noise MSE and both unclipped and clipped clean-image
MSE at explicitly requested timesteps.

This is a one-step diagnostic. It is not a full reverse-trajectory result,
does not validate generated images, and does not establish a Stage 0 result.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from exposure.denoise_probe import reconstruct_x0
from exposure.pilot_train import PilotUNet, encode_rules, load_rules, schedule
from exposure.reconstruction_probe import load_target


PREDICTORS = ("raw", "ema", "copy", "zero", "oracle")


def _unique_ints(name: str, values: object, *, minimum: int, maximum: int | None = None) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list of integers")
    if any(type(value) is not int for value in values):
        raise ValueError(f"{name} must contain integers only")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values")
    if any(value < minimum for value in values):
        raise ValueError(f"{name} values must be at least {minimum}")
    if maximum is not None and any(value > maximum for value in values):
        raise ValueError(f"{name} values must be at most {maximum}")
    return values


def _checkpoint_parts(checkpoint: object) -> tuple[dict, int, int]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    for key in ("config", "rules", "model", "ema"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint lacks required key: {key!r}")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary")
    timesteps = config.get("timesteps")
    width = config.get("width")
    if type(timesteps) is not int or timesteps < 2:
        raise ValueError("Checkpoint timesteps must be an integer at least 2")
    if type(width) is not int or width < 4 or width % 4:
        raise ValueError("Checkpoint width must be a positive multiple of four, at least four")
    if not isinstance(checkpoint["model"], dict) or not isinstance(checkpoint["ema"], dict):
        raise ValueError("Checkpoint model and EMA state must be dictionaries")
    return config, timesteps, width


def _metrics(predicted_noise: torch.Tensor, noise: torch.Tensor, noisy: torch.Tensor,
             clean: torch.Tensor, alpha_bar: torch.Tensor) -> dict[str, float]:
    reconstructed = reconstruct_x0(noisy, predicted_noise, alpha_bar)
    clipped = reconstructed.clamp(-1.0, 1.0)
    values = {
        "noise_mse": float((predicted_noise - noise).square().mean()),
        "x0_mse": float((reconstructed - clean).square().mean()),
        "x0_clipped_mse": float((clipped - clean).square().mean()),
        # Ignore small floating-point reconstruction errors at the boundary.
        "x0_outside_fraction": float((reconstructed.abs() > 1.0 + 1e-4).to(torch.float32).mean()),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite high-noise baseline result")
    return values


@torch.no_grad()
def diagnose(
    checkpoint: dict,
    target: Any,
    *,
    timesteps: list[int],
    seeds: list[int],
    rule: str | None = None,
    checkpoint_path: Path | str | None = None,
    target_path: Path | str | None = None,
) -> dict[str, Any]:
    """Audit one-step high-noise predictors on a fixed CPU target and matched noise."""
    config, total_timesteps, width = _checkpoint_parts(checkpoint)
    timesteps = _unique_ints("timesteps", timesteps, minimum=0, maximum=total_timesteps - 1)
    seeds = _unique_ints("seeds", seeds, minimum=0)

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
        model = PilotUNet(width, total_timesteps).to("cpu")
        try:
            model.load_state_dict(checkpoint[name])
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid checkpoint {name} state: {exc}") from exc
        models[name] = model.eval()

    _, alpha_bar = schedule(total_timesteps, torch.device("cpu"))

    results = []
    for timestep in timesteps:
        alpha = alpha_bar[timestep]
        time = torch.tensor([timestep], dtype=torch.long)
        seed_results = []
        for seed in seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(clean.shape, generator=generator, device="cpu")
            noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

            predictions = {
                "raw": models["model"](noisy, time, tokens),
                "ema": models["ema"](noisy, time, tokens),
                "copy": noisy,
                "zero": torch.zeros_like(noise),
                "oracle": noise,
            }
            seed_results.append({
                "seed": seed,
                "predictors": {
                    name: _metrics(predictions[name], noise, noisy, clean, alpha)
                    for name in PREDICTORS
                },
            })

        averages = {
            name: {
                metric: sum(seed_result["predictors"][name][metric] for seed_result in seed_results)
                / len(seed_results)
                for metric in ("noise_mse", "x0_mse", "x0_clipped_mse", "x0_outside_fraction")
            }
            for name in PREDICTORS
        }
        results.append({"timestep": timestep, "seeds": seed_results, "averages": averages})

    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "target": target_display,
        "rule": rule,
        "schedule_timesteps": total_timesteps,
        "timesteps": timesteps,
        "seeds": seeds,
        "checkpoint_steps": config.get("steps"),
        "predictors": list(PREDICTORS),
        "results": results,
        "note": (
            "One-step high-noise baseline audit on CPU with one fixed target and matched noise; "
            "reports noise MSE and unclipped/clipped x0 MSE. This is not a full reverse-trajectory "
            "measurement, does not validate generated images, and does not establish a Stage 0 result. "
            "The outside fraction uses a 1e-4 tolerance beyond [-1, 1] for floating-point error."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step high-noise baseline audit for pilot checkpoints")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--rule", type=str, default=None, help="rule name (required if checkpoint has multiple rules)")
    parser.add_argument("--timesteps", "--timestep", dest="timesteps", type=int, nargs="+",
                        default=[900, 950, 990, 999], help="one-step timesteps in [0, schedule-1]")
    parser.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=[0],
                        help="one or more unique nonnegative random seeds")
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
            checkpoint_path=args.checkpoint,
            target_path=args.target,
        )
        content = json.dumps(report, indent=2, allow_nan=False) + "\n"

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                file.write(content)
        else:
            print(content, end="")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
