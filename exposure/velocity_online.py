"""Exploratory online conditional velocity pilot; not a Stage 0 result.

Draws new rendered scenes and records requested and realized rule counts.
Checkpoints require this module's velocity sampler. Version 1 checkpoints can
resume with the exact code and runtime; earlier schema 0 cannot be resumed.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from exposure import identity_eps_finetune, pilot_train, velocity_probe
from exposure.identity_eps_finetune import _build_model, _sample
from exposure.measure import unique_object
from exposure.render import render_conditioned
from exposure.rules import satisfies
from exposure.velocity_probe import VelocityAsNoise, velocity_target


def _runtime(device):
    runtime = pilot_train._runtime(device)
    for name, path in (("velocity_online", Path(__file__)),
                       ("velocity_probe", Path(velocity_probe.__file__)),
                       ("identity_eps_finetune", Path(identity_eps_finetune.__file__))):
        runtime["code_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return runtime


def _resume_parts(checkpoint):
    required = {"schema_version", "diagnostic", "config", "rules", "model", "ema", "optimizer", "report", "rng"}
    if (not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1
            or not required <= checkpoint.keys() or not isinstance(checkpoint["diagnostic"], dict)
            or checkpoint["diagnostic"].get("kind") != "online_velocity"
            or checkpoint["diagnostic"].get("prediction") != "velocity"
            or checkpoint["diagnostic"].get("requires_custom_sampler") is not True
            or checkpoint["diagnostic"].get("resumable") is not True):
        raise ValueError("Resuming requires an online velocity schema 1 checkpoint; schema 0 is not resumable")
    config, report, rng = checkpoint["config"], checkpoint["report"], checkpoint["rng"]
    config_keys = {"steps", "batch_size", "seed", "device", "width", "timesteps", "lr", "dropout",
                   "ema_decay", "max_attempts", "deterministic"}
    report_keys = {"weights", "condition_counts", "realized_rule_counts_by_condition", "sampling_attempts",
                   "elapsed_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"}
    if (not isinstance(config, dict) or not config_keys <= config.keys()
            or not isinstance(report, dict) or not report_keys <= report.keys()
            or not isinstance(rng, dict) or set(rng) != {"runtime", "python", "torch_cpu", "torch_cuda"}
            or type(config["steps"]) is not int or config["steps"] < 1):
        raise ValueError("Checkpoint lacks required resume state")
    return config, report, rng


def train(definitions, weights=None, *, steps, batch_size, seed, device, width, timesteps,
          lr, dropout, ema_decay, max_attempts, deterministic, resume=None):
    """Fit velocity prediction on fresh conditional renders and report exposure."""
    for name, value, lower in (("steps", steps, 1), ("batch_size", batch_size, 1),
                               ("seed", seed, 0), ("timesteps", timesteps, 2),
                               ("max_attempts", max_attempts, 1)):
        if type(value) is not int or value < lower:
            raise ValueError(f"{name} must be an integer at least {lower}")
    if device not in ("cpu", "cuda") or type(deterministic) is not bool:
        raise ValueError("Invalid device or determinism setting")
    for name, value, lo, hi in (("lr", lr, 0, None), ("dropout", dropout, -1, 1), ("ema_decay", ema_decay, -1, 1)):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value <= lo or (hi is not None and value >= hi)):
            raise ValueError(f"Invalid {name}")
    if dropout < 0 or ema_decay < 0:
        raise ValueError("dropout and ema_decay must be nonnegative")
    rules = pilot_train.load_rules(definitions)
    names = list(rules)
    if not names:
        raise ValueError("At least one rule is required")
    probs = pilot_train._weights(names, weights)
    pilot_train._configure_determinism(deterministic, device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    target = torch.device(device)
    runtime = _runtime(device)
    previous_steps, previous_elapsed, previous_allocated, previous_reserved = 0, 0.0, None, None
    previous_report = previous_rng = None
    if resume is not None:
        old, previous_report, previous_rng = _resume_parts(resume)
        expected = {"batch_size": batch_size, "seed": seed, "device": device, "width": width,
                    "timesteps": timesteps, "lr": lr, "dropout": dropout, "ema_decay": ema_decay,
                    "max_attempts": max_attempts, "deterministic": deterministic}
        if (any(old[k] != value for k, value in expected.items()) or resume["rules"] != definitions
                or list(resume["rules"]) != names or previous_rng["runtime"] != runtime
                or pilot_train._weights(names, previous_report["weights"]) != probs):
            raise ValueError("Incompatible velocity resume configuration, rules, weights, device or runtime")
        previous_steps = old["steps"]
        previous_elapsed = previous_report["elapsed_seconds"]
        previous_allocated = previous_report["peak_cuda_allocated_bytes"]
        previous_reserved = previous_report["peak_cuda_reserved_bytes"]
        if (isinstance(previous_elapsed, bool) or not isinstance(previous_elapsed, (int, float))
                or not math.isfinite(previous_elapsed) or previous_elapsed < 0
                or (device == "cpu" and (previous_allocated is not None or previous_reserved is not None))
                or (device == "cuda" and any(type(v) is not int or v < 0
                                             for v in (previous_allocated, previous_reserved)))):
            raise ValueError("Invalid checkpoint timing or memory measurements")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = _build_model(width, timesteps, target, seed)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    tokens = pilot_train.encode_rules(rules).to(target)
    _, alpha_bar = pilot_train.schedule(timesteps, target)
    if resume is None:
        counts = {name: 0 for name in names}
        realized = {name: {event: 0 for event in names} for name in names}
        attempts_total = 0
    else:
        counts = copy.deepcopy(previous_report["condition_counts"])
        realized = copy.deepcopy(previous_report["realized_rule_counts_by_condition"])
        attempts_total = previous_report["sampling_attempts"]
        if (not isinstance(counts, dict) or set(counts) != set(names)
                or any(type(v) is not int or v < 0 for v in counts.values())
                or sum(counts.values()) != previous_steps * batch_size
                or not isinstance(realized, dict) or set(realized) != set(names)
                or any(not isinstance(row, dict) or set(row) != set(names)
                       or any(type(v) is not int or v < 0 or v > counts[name] for v in row.values())
                       for name, row in realized.items())
                or type(attempts_total) is not int or attempts_total < sum(counts.values())):
            raise ValueError("Invalid accumulated exposure counts")
        try:
            model.load_state_dict(resume["model"])
            ema.load_state_dict(resume["ema"])
            optimizer.load_state_dict(copy.deepcopy(resume["optimizer"]))
            rng.setstate(previous_rng["python"])
            torch.set_rng_state(previous_rng["torch_cpu"])
            if device == "cuda":
                states = previous_rng["torch_cuda"]
                if not isinstance(states, list) or len(states) != torch.cuda.device_count():
                    raise ValueError("CUDA RNG state does not match visible devices")
                torch.cuda.set_rng_state_all(states)
            elif previous_rng["torch_cuda"] is not None:
                raise ValueError("CPU checkpoint contains CUDA RNG state")
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid checkpoint model, optimizer or RNG state: {exc}") from exc
    if device == "cuda":
        torch.cuda.synchronize(target)
        torch.cuda.reset_peak_memory_stats(target)
    start = time.perf_counter()
    for _ in range(steps):
        indices = rng.choices(range(len(names)), weights=probs, k=batch_size)
        batch = []
        for index in indices:
            name = names[index]
            scene, attempts = render_conditioned(rng.getrandbits(63), rules[name], max_attempts)
            counts[name] += 1
            attempts_total += attempts
            for event, predicates in rules.items():
                realized[name][event] += int(satisfies(scene.objects, predicates))
            pixels = np.array(scene.image, dtype=np.float32, copy=True)
            batch.append(torch.from_numpy(pixels).permute(2, 0, 1) / 127.5 - 1.0)
        clean = torch.stack(batch).to(target)
        t = torch.randint(timesteps, (batch_size,), device=target)
        noise = torch.randn_like(clean)
        a = alpha_bar[t, None, None, None]
        noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
        drop = torch.rand(batch_size, device=target) < dropout
        loss = F.mse_loss(model(noisy, t, tokens[indices], drop), velocity_target(clean, noise, a))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite velocity training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                average.lerp_(parameter, 1 - ema_decay)
    if device == "cuda":
        torch.cuda.synchronize(target)
    config = {"steps": previous_steps + steps, "batch_size": batch_size, "seed": seed, "device": device,
              "width": width, "timesteps": timesteps, "lr": lr, "dropout": dropout,
              "ema_decay": ema_decay, "max_attempts": max_attempts, "deterministic": deterministic}
    report = {"config": config, "condition_counts": counts, "realized_rule_counts_by_condition": realized,
              "weights": dict(zip(names, probs, strict=True)), "sampling_attempts": attempts_total,
              "last_velocity_loss": float(loss.detach().cpu()), "elapsed_seconds": previous_elapsed + time.perf_counter() - start,
              "peak_cuda_allocated_bytes": max(previous_allocated or 0, torch.cuda.max_memory_allocated(target)) if device == "cuda" else None,
              "peak_cuda_reserved_bytes": max(previous_reserved or 0, torch.cuda.max_memory_reserved(target)) if device == "cuda" else None,
              "note": "Online instrument smoke only, not model validation, research evidence or Stage 0 passage."}
    checkpoint = {"schema_version": 1, "config": config, "rules": definitions,
                  "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "ema": {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()},
                  "optimizer": optimizer.state_dict(), "report": report,
                  "rng": {"runtime": runtime, "python": rng.getstate(), "torch_cpu": torch.get_rng_state(),
                          "torch_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None},
                  "diagnostic": {"kind": "online_velocity", "prediction": "velocity",
                                 "requires_custom_sampler": True, "resumable": True,
                                 "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    return checkpoint, report


def sample(checkpoint, rule, *, seed, device):
    """Use the DDPM sampler with velocity-to-epsilon conversion and EMA weights."""
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("diagnostic"), dict)
            or checkpoint["diagnostic"].get("kind") != "online_velocity"
            or checkpoint["diagnostic"].get("prediction") != "velocity"):
        raise ValueError("Expected an online velocity checkpoint")
    if type(seed) is not int or seed < 0 or device not in ("cpu", "cuda"):
        raise ValueError("Invalid seed or device")
    config = checkpoint["config"]
    pilot_train._configure_determinism(config["deterministic"], device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    target = torch.device(device)
    rules = pilot_train.load_rules(checkpoint["rules"])
    names = list(rules)
    if rule not in rules:
        raise ValueError("Unknown rule")
    model = pilot_train.PilotUNet(config["width"], config["timesteps"]).to(target)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    tokens = pilot_train.encode_rules(rules)[names.index(rule):names.index(rule) + 1].to(target)
    beta, alpha_bar = pilot_train.schedule(config["timesteps"], target)
    return _sample(VelocityAsNoise(model, alpha_bar), "standard", tokens=tokens, beta=beta,
                   alpha_bar=alpha_bar, seed=seed, device=target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--rules", type=Path, required=True)
    fit.add_argument("--weights", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--steps", type=int, default=100)
    fit.add_argument("--batch-size", type=int, default=2)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    fit.add_argument("--width", type=int, default=16)
    fit.add_argument("--timesteps", type=int, default=1000)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--dropout", type=float, default=0.1)
    fit.add_argument("--ema-decay", type=float, default=0.999)
    fit.add_argument("--max-attempts", type=int, default=10000)
    fit.add_argument("--deterministic", action="store_true")
    resume = commands.add_parser("resume", help="continue a compatible schema 1 velocity checkpoint")
    resume.add_argument("--checkpoint", type=Path, required=True)
    resume.add_argument("--steps", type=int, required=True)
    resume.add_argument("--output", type=Path, required=True)
    generate = commands.add_parser("sample")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--rule", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists")
        if args.command == "train":
            definitions = json.loads(args.rules.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
            weights = json.loads(args.weights.read_text(encoding="utf-8"), object_pairs_hook=unique_object) if args.weights else None
            checkpoint, report = train(definitions, weights, steps=args.steps, batch_size=args.batch_size,
                seed=args.seed, device=args.device, width=args.width, timesteps=args.timesteps, lr=args.lr,
                dropout=args.dropout, ema_decay=args.ema_decay, max_attempts=args.max_attempts,
                deterministic=args.deterministic)
            pilot_train.save_checkpoint(args.output, checkpoint)
            print(json.dumps(report, indent=2, allow_nan=False))
        elif args.command == "resume":
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            config, report, _ = _resume_parts(checkpoint)
            resumed, result = train(checkpoint["rules"], report["weights"], steps=args.steps,
                batch_size=config["batch_size"], seed=config["seed"], device=config["device"],
                width=config["width"], timesteps=config["timesteps"], lr=config["lr"],
                dropout=config["dropout"], ema_decay=config["ema_decay"],
                max_attempts=config["max_attempts"], deterministic=config["deterministic"], resume=checkpoint)
            pilot_train.save_checkpoint(args.output, resumed)
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            image = sample(checkpoint, args.rule, seed=args.seed, device=args.device)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as file:
                image.save(file, format="PNG")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
