"""Matched classifier-free guidance sweep for an online velocity checkpoint.

Images are exploratory outputs, not human labels or evidence of a stage gate.
Each scale for a given rule and seed starts with the same random stream.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from exposure import pilot_train
from exposure.identity_eps_finetune import _sample
from exposure.velocity_probe import velocity_to_noise


class GuidedVelocity(torch.nn.Module):
    """Convert guided velocity predictions to epsilon for the DDPM sampler."""

    def __init__(self, model, alpha_bar, scale):
        super().__init__()
        self.model = model
        self.register_buffer("alpha_bar", alpha_bar)
        self.scale = scale

    def forward(self, noisy, timestep, tokens, drop=None):
        if self.scale == 1.0:
            velocity = self.model(noisy, timestep, tokens)
        elif self.scale == 0.0:
            velocity = self.model(noisy, timestep, tokens, torch.ones(
                noisy.shape[0], dtype=torch.bool, device=noisy.device))
        else:
            conditional = self.model(noisy, timestep, tokens)
            unconditional = self.model(noisy, timestep, tokens, torch.ones(
                noisy.shape[0], dtype=torch.bool, device=noisy.device))
            velocity = unconditional + self.scale * (conditional - unconditional)
        alpha = self.alpha_bar[timestep].view(-1, 1, 1, 1)
        return velocity_to_noise(noisy, velocity, alpha)


def sweep(checkpoint, output, *, rules, seeds, scales, device, checkpoint_path=None):
    """Generate matched samples without overwriting an existing directory."""
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory already exists")
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("diagnostic"), dict)
            or checkpoint["diagnostic"].get("kind") != "online_velocity"
            or checkpoint["diagnostic"].get("prediction") != "velocity"
            or checkpoint["diagnostic"].get("requires_custom_sampler") is not True):
        raise ValueError("Expected a marked online velocity checkpoint")
    config = checkpoint.get("config")
    if (not isinstance(config, dict) or type(config.get("deterministic")) is not bool
            or type(config.get("timesteps")) is not int or config["timesteps"] < 2
            or type(config.get("width")) is not int):
        raise ValueError("Invalid checkpoint configuration")
    if device not in ("cpu", "cuda"):
        raise ValueError("Invalid device")
    definitions = pilot_train.load_rules(checkpoint["rules"])
    if (not isinstance(rules, list) or not rules or any(not isinstance(x, str) or x not in definitions for x in rules)
            or len(set(rules)) != len(rules)):
        raise ValueError("Rules must be distinct names from the checkpoint")
    if (not isinstance(seeds, list) or not seeds or any(type(x) is not int or x < 0 for x in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Seeds must be unique nonnegative integers")
    if (not isinstance(scales, list) or not scales
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x < 0 for x in scales)
            or len(set(scales)) != len(scales)):
        raise ValueError("Scales must be unique finite nonnegative numbers")

    pilot_train._configure_determinism(config["deterministic"], device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    target = torch.device(device)
    model = pilot_train.PilotUNet(config["width"], config["timesteps"]).to(target)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    all_tokens = pilot_train.encode_rules(definitions).to(target)
    names = list(definitions)
    beta, alpha_bar = pilot_train.schedule(config["timesteps"], target)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for rule_index, rule in enumerate(rules):
        index = names.index(rule)
        tokens = all_tokens[index:index + 1]
        for seed in seeds:
            for scale_index, scale in enumerate(scales):
                predictor = GuidedVelocity(model, alpha_bar, float(scale))
                image = _sample(predictor, "standard", tokens=tokens, beta=beta,
                                alpha_bar=alpha_bar, seed=seed, device=target)
                path = output / f"rule-{rule_index}-seed-{seed}-scale-{scale_index}.png"
                with path.open("xb") as file:
                    image.save(file, format="PNG")
                rows.append({"rule": rule, "seed": seed, "scale": float(scale), "path": str(path)})
    report = {
        "schema_version": 1, "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "training_steps": config.get("steps"), "training_dropout": config.get("dropout"),
        "device": device, "deterministic": config["deterministic"],
        "rules": rules, "seeds": seeds, "scales": [float(x) for x in scales], "samples": rows,
        "note": ("Same rule and seed use matched initial and transition random streams across scales. "
                 "Exploratory images only; no human labels, quality metric or Stage 0 result. "
                 "Classifier-free guidance depends on conditional dropout during training."),
    }
    if checkpoint_path:
        report["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    with (output / "report.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--rules", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--scales", nargs="+", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output directory already exists")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        report = sweep(checkpoint, args.output, rules=args.rules, seeds=args.seeds,
                       scales=args.scales, device=args.device, checkpoint_path=args.checkpoint)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
