"""Matched fixed-scene velocity training with fresh AdamW and momentum SGD.

Both optimizers start from the same freshly initialized network and empty state.
This is exploratory, not an optimizer recommendation or Stage 0 result.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from exposure import pilot_train
from exposure.identity_eps_finetune import _build_model, _model_hash, _one_step_metrics, _sample, _unique_ints
from exposure.measure import unique_object
from exposure.reconstruction_probe import load_target
from exposure.velocity_probe import VelocityAsNoise, velocity_target


def run(definitions, target, output, *, rule, steps, batch_size, seed, init_seed,
        width, timesteps, adamw_lr, sgd_lr, device, deterministic, eval_times,
        eval_seeds, sample_seeds):
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory already exists")
    if any(type(x) is not int or x < 1 for x in (steps, batch_size)):
        raise ValueError("steps and batch size must be positive integers")
    if any(type(x) is not int or x < 0 for x in (seed, init_seed)):
        raise ValueError("seeds must be nonnegative integers")
    if (type(timesteps) is not int or timesteps < 2 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
                for x in (adamw_lr, sgd_lr))):
        raise ValueError("Invalid schedule or learning rate")
    if device not in ("cpu", "cuda") or type(deterministic) is not bool:
        raise ValueError("Invalid device or deterministic setting")
    eval_times = _unique_ints("eval_times", eval_times, minimum=0, maximum=timesteps - 1)
    eval_seeds = _unique_ints("eval_seeds", eval_seeds, minimum=0)
    sample_seeds = _unique_ints("sample_seeds", sample_seeds, minimum=0)
    rules = pilot_train.load_rules(definitions)
    if rule not in rules:
        raise ValueError("Unknown rule")
    pilot_train._configure_determinism(deterministic, device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    dev = torch.device(device)
    clean, _ = load_target(target)
    clean = clean.to(dev)
    names = list(rules)
    index = names.index(rule)
    tokens = pilot_train.encode_rules(rules)[index:index + 1].to(dev)
    beta, alpha_bar = pilot_train.schedule(timesteps, dev)
    target_np = (clean[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
    arms, hashes = {}, {}
    output.mkdir(parents=True, exist_ok=False)
    for name, lr in (("adamw", adamw_lr), ("sgd", sgd_lr)):
        model = _build_model(width, timesteps, dev, init_seed)
        hashes[name] = _model_hash(model)
        ema = copy.deepcopy(model).eval()
        # Equal weight decay, but AdamW decouples it while SGD couples it.
        optimizer = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                     if name == "adamw" else
                     torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01))
        time_gen = torch.Generator(device=dev).manual_seed(seed)
        noise_gen = torch.Generator(device=dev).manual_seed(seed + 7919)
        batch = clean.repeat(batch_size, 1, 1, 1)
        batch_tokens = tokens.repeat(batch_size, 1, 1)
        counts = [0] * timesteps
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        start = time.perf_counter()
        for _ in range(steps):
            t = torch.randint(timesteps, (batch_size,), generator=time_gen, device=dev)
            noise = torch.randn(batch.shape, generator=noise_gen, device=dev)
            a = alpha_bar[t, None, None, None]
            noisy = a.sqrt() * batch + (1 - a).sqrt() * noise
            loss = F.mse_loss(model(noisy, t, batch_tokens), velocity_target(batch, noise, a))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                    average.lerp_(parameter, 0.001)
            for step in t.detach().cpu().tolist():
                counts[step] += 1
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
        nets = {"raw": VelocityAsNoise(model, alpha_bar), "ema": VelocityAsNoise(ema, alpha_bar)}
        metrics = {key: _one_step_metrics(net, "standard", clean=clean, tokens=tokens,
                    alpha_bar=alpha_bar, timesteps=eval_times, seeds=eval_seeds, device=dev)
                   for key, net in nets.items()}
        samples = []
        for sample_seed in sample_seeds:
            image = _sample(nets["ema"], "standard", tokens=tokens, beta=beta, alpha_bar=alpha_bar,
                            seed=sample_seed, device=dev)
            path = output / f"{name}-{sample_seed}.png"
            image.save(path)
            mse = float(np.mean((np.asarray(image, dtype=np.float32) / 255 - target_np) ** 2))
            samples.append({"seed": sample_seed, "path": str(path), "mse_0_1": mse})
        checkpoint = {"schema_version": 0, "config": {"width": width, "timesteps": timesteps},
                      "rules": definitions, "model": {k: v.cpu() for k, v in model.state_dict().items()},
                      "ema": {k: v.cpu() for k, v in ema.state_dict().items()},
                      "diagnostic": {"kind": "optimizer_probe", "prediction": "velocity",
                                     "optimizer": name, "resumable": False, "requires_custom_sampler": True}}
        pilot_train.save_checkpoint(output / f"{name}.pt", checkpoint)
        arms[name] = {"lr": lr, "metrics": metrics, "samples": samples, "elapsed_seconds": elapsed,
                      "peak_cuda_allocated_bytes": peak, "last_objective_loss": float(loss.detach().cpu()),
                      "timestep_counts": {str(i): n for i, n in enumerate(counts) if n}}
    report = {"schema_version": 1, "rule": rule, "target": str(target),
              "target_sha256": hashlib.sha256(Path(target).read_bytes()).hexdigest(),
              "initial_model_sha256": hashes,
              "config": {"steps": steps, "batch_size": batch_size, "seed": seed, "init_seed": init_seed,
                         "width": width, "timesteps": timesteps, "device": device, "deterministic": deterministic,
                         "weight_decay": 0.01, "sgd_momentum": 0.9, "dropout": 0.0, "ema_decay": 0.999,
                         "adamw_lr": adamw_lr, "sgd_lr": sgd_lr, "eval_times": eval_times,
                         "eval_seeds": eval_seeds, "sample_seeds": sample_seeds},
              "arms": arms,
              "constant_target_mean_mse_0_1": float(np.mean((target_np - target_np.mean(axis=(0, 1))) ** 2)),
              "note": ("Fresh optimizer states and identical initial weights, target and random streams. "
                       "SGD and AdamW weight decay differ. Learning rates are not tuned; this single-scene "
                       "diagnostic cannot establish an optimizer winner, generalization or Stage 0 passage.")}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--init-seed", type=int, default=46)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--adamw-lr", type=float, default=1e-4)
    parser.add_argument("--sgd-lr", type=float, default=0.01)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-times", nargs="+", type=int, default=[20, 100, 500, 900, 999])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sample-seeds", nargs="+", type=int, default=[5, 6])
    args = parser.parse_args()
    try:
        definitions = json.loads(args.rules.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        report = run(definitions, args.target, args.output, rule=args.rule, steps=args.steps,
                     batch_size=args.batch_size, seed=args.seed, init_seed=args.init_seed, width=args.width,
                     timesteps=args.timesteps, adamw_lr=args.adamw_lr, sgd_lr=args.sgd_lr,
                     device=args.device, deterministic=args.deterministic, eval_times=args.eval_times,
                     eval_seeds=args.eval_seeds, sample_seeds=args.sample_seeds)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
