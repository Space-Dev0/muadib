"""Compare matched denoising losses under correct, false and null conditions.

This is a teacher-forced diagnostic on freshly rendered annotated scenes, not
an evaluation of generated samples or a Stage 0 gate.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from exposure import pilot_train
from exposure.render import render_conditioned
from exposure.rules import satisfies
from exposure.velocity_probe import velocity_target


@torch.no_grad()
def diagnose(checkpoint, *, scene_seeds, noise_seed, timesteps, device="cpu", max_attempts=10000):
    """Evaluate each rule on fresh renders with matched noise and false rules."""
    marker = checkpoint.get("diagnostic", {}) if isinstance(checkpoint, dict) else {}
    if (not isinstance(marker, dict) or marker.get("kind") != "online_velocity"
            or marker.get("prediction") != "velocity" or marker.get("requires_custom_sampler") is not True):
        raise ValueError("Expected an online velocity checkpoint")
    config = checkpoint["config"]
    total = config["timesteps"]
    if (not isinstance(scene_seeds, list) or not scene_seeds
            or any(type(x) is not int or x < 0 for x in scene_seeds)
            or len(set(scene_seeds)) != len(scene_seeds)):
        raise ValueError("scene_seeds must be unique nonnegative integers")
    if (type(noise_seed) is not int or noise_seed < 0 or type(max_attempts) is not int or max_attempts < 1
            or not isinstance(timesteps, list) or not timesteps
            or any(type(t) is not int or not 0 <= t < total for t in timesteps)
            or len(set(timesteps)) != len(timesteps) or device not in ("cpu", "cuda")):
        raise ValueError("Invalid noise seed, timesteps, device or attempts")
    definitions = pilot_train.load_rules(checkpoint["rules"])
    names = list(definitions)
    if len(names) < 2:
        raise ValueError("At least two rules are needed for false-condition comparisons")
    pilot_train._configure_determinism(config["deterministic"], device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    target = torch.device(device)
    model = pilot_train.PilotUNet(config["width"], total).to(target)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    tokens = pilot_train.encode_rules(definitions).to(target)
    _, alpha_bar = pilot_train.schedule(total, target)
    generator = torch.Generator(device=target).manual_seed(noise_seed)
    rows = []
    excluded = []
    for index, name in enumerate(names):
        for scene_seed in scene_seeds:
            scene, attempts = render_conditioned(scene_seed, definitions[name], max_attempts)
            wrong = [other for other in names if other != name and not satisfies(scene.objects, definitions[other])]
            if not wrong:
                excluded.append({"rule": name, "scene_seed": scene_seed, "reason": "no_false_rule"})
                continue
            # Rotate the first candidate by scene seed, then keep the order stable.
            other = wrong[scene_seed % len(wrong)]
            pixels = np.array(scene.image, dtype=np.float32, copy=True)
            clean = (torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).to(target)
            for step in timesteps:
                noise = torch.randn(clean.shape, generator=generator, device=target)
                a = alpha_bar[step]
                noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
                expected = velocity_target(clean, noise, a)
                t = torch.tensor([step], device=target)
                right = tokens[index:index + 1]
                bad_index = names.index(other)
                bad = tokens[bad_index:bad_index + 1]
                null = torch.ones(1, dtype=torch.bool, device=target)
                losses = {
                    "correct": float(F.mse_loss(model(noisy, t, right), expected).cpu()),
                    "false": float(F.mse_loss(model(noisy, t, bad), expected).cpu()),
                    "null": float(F.mse_loss(model(noisy, t, right, null), expected).cpu()),
                }
                if not all(math.isfinite(v) for v in losses.values()):
                    raise ValueError("Non-finite loss")
                rows.append({"rule": name, "false_rule": other, "scene_seed": scene_seed,
                             "timestep": step, "attempts": attempts, "loss": losses,
                             "false_minus_correct": losses["false"] - losses["correct"],
                             "null_minus_correct": losses["null"] - losses["correct"]})
    summary = []
    for name in names:
        for step in timesteps:
            matched = [row for row in rows if row["rule"] == name and row["timestep"] == step]
            summary.append({"rule": name, "timestep": step, "count": len(matched),
                            **{key: (sum(x[key] for x in matched) / len(matched) if matched else None)
                               for key in ("false_minus_correct", "null_minus_correct")}})
    return {"schema_version": 1, "training_steps": config.get("steps"), "device": device,
            "scene_seeds": scene_seeds, "noise_seed": noise_seed, "timesteps": timesteps,
            "rows": rows, "excluded": excluded, "summary": summary,
            "note": ("Teacher-forced paired velocity losses on fresh annotated renders. False rules are "
                     "verified false for each scene; positive differences favor the correct prompt. "
                     "Not generated-image accuracy, causal evidence, or a Stage 0 gate. Distinct seeds "
                     "do not guarantee disjoint scenes; no uncertainty interval is calculated.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--timesteps", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output path already exists")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        report = diagnose(checkpoint, scene_seeds=args.scene_seeds, noise_seed=args.noise_seed,
                          timesteps=args.timesteps, device=args.device)
        report["checkpoint"] = str(args.checkpoint)
        report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
