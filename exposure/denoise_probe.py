"""Compare raw and EMA one-step denoising on matched rendered scenes.

This diagnostic measures noise prediction and reconstruction errors; it does
not validate the full sampler, generated images, or any research gate.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from exposure.pilot_train import PilotUNet, encode_rules, load_rules, schedule
from exposure.render import render_conditioned


def reconstruct_x0(noisy: torch.Tensor, predicted_noise: torch.Tensor,
                   alpha_bar: torch.Tensor) -> torch.Tensor:
    """Invert the forward noising equation for a predicted noise tensor."""
    return (noisy - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()


@torch.no_grad()
def diagnose(checkpoint: dict, *, rule: str, samples: int, timesteps: list[int], seed: int) -> dict:
    """Evaluate both weight sets on identical CPU scenes and noise draws."""
    if type(samples) is not int or samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("samples must be positive and seed must be nonnegative")
    config = checkpoint["config"]
    if (not isinstance(timesteps, list) or not timesteps or
            any(type(t) is not int or t < 0 or t >= config["timesteps"] for t in timesteps) or
            len(set(timesteps)) != len(timesteps)):
        raise ValueError("timesteps must be unique indices inside the checkpoint schedule")
    rules = load_rules(checkpoint["rules"])
    if rule not in rules:
        raise ValueError("Unknown rule")
    names = list(rules)
    index = names.index(rule)
    tokens = encode_rules(rules)[index:index + 1]
    models = {}
    for name in ("model", "ema"):
        model = PilotUNet(config["width"], config["timesteps"])
        model.load_state_dict(checkpoint[name])
        models[name] = model.eval()
    _, alpha_bar = schedule(config["timesteps"], torch.device("cpu"))
    rng = random.Random(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scenes = []
    for _ in range(samples):
        scene, _ = render_conditioned(rng.getrandbits(63), rules[rule], config["max_attempts"])
        pixels = np.array(scene.image, dtype=np.float32, copy=True)
        scenes.append(torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1)
    result = []
    for timestep in timesteps:
        totals = {name: {"noise_mse": 0.0, "x0_mse": 0.0} for name in ("model", "ema", "zero")}
        abar = alpha_bar[timestep]
        t = torch.tensor([timestep])
        for clean in scenes:
            noise = torch.randn(clean.shape, generator=generator)
            noisy = abar.sqrt() * clean + (1 - abar).sqrt() * noise
            estimates = {name: model(noisy, t, tokens) for name, model in models.items()}
            estimates["zero"] = torch.zeros_like(noise)
            for name, prediction in estimates.items():
                restored = reconstruct_x0(noisy, prediction, abar)
                totals[name]["noise_mse"] += float((prediction - noise).square().mean())
                totals[name]["x0_mse"] += float((restored - clean).square().mean())
        averages = {name: {metric: value / samples for metric, value in metrics.items()}
                    for name, metrics in totals.items()}
        if not all(math.isfinite(value) for metrics in averages.values() for value in metrics.values()):
            raise ValueError("Non-finite diagnostic result")
        result.append({"timestep": timestep, "metrics": averages})
    return {"schema_version": 1, "rule": rule, "samples": samples, "seed": seed,
            "checkpoint_steps": config["steps"], "results": result,
            "note": "One-step CPU diagnostic on matched renders; not a sampler or Stage 0 validation."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw and EMA one-step denoising")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[20, 100, 500])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional new JSON output; never overwritten")
    args = parser.parse_args()
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        result = diagnose(checkpoint, rule=args.rule, samples=args.samples,
                          timesteps=args.timesteps, seed=args.seed)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                file.write(content)
        else:
            print(content, end="")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
