"""Small paired high-noise fine-tuning experiment, not a research result.

This module starts from an existing pilot or fixed-scene checkpoint and runs
two arms with identical initial weights, optimizer state, target, timesteps,
noise, and dropout masks. The control arm uses the ordinary per-sample MSE.
The treatment arm multiplies the loss for timesteps at or above a cutoff by a
fixed weight, normalized by the expected weight under uniform timestep
sampling so the full-schedule mean weight is one.

The full diffusion schedule is retained. No existing checkpoint schema or
``pilot_train.py`` behavior is changed. The saved arm checkpoints are
non-resumable diagnostics (schema version 0).
"""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from exposure import high_t_baseline, pilot_train
from exposure.pilot_train import PilotUNet, encode_rules, load_rules, schedule
from exposure.reconstruction_probe import load_target


def _validate_int(name: str, value: object, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def loss_weights(timesteps: torch.Tensor, *, high_t_min: int, high_t_weight: float,
                 total_timesteps: int) -> torch.Tensor:
    """Return importance weights with mean one under uniform timestep sampling."""
    if type(high_t_min) is not int or high_t_min < 0:
        raise ValueError("high_t_min must be a nonnegative integer")
    if type(total_timesteps) is not int or total_timesteps < 2 or high_t_min >= total_timesteps:
        raise ValueError("total_timesteps must be at least two and greater than high_t_min")
    if not math.isfinite(high_t_weight) or high_t_weight < 1.0:
        raise ValueError("high_t_weight must be finite and at least one")
    weights = torch.ones_like(timesteps, dtype=torch.float32)
    if high_t_weight > 1.0:
        high = torch.full_like(weights, float(high_t_weight))
        weights = torch.where(timesteps >= high_t_min, high, weights)
    high_fraction = (total_timesteps - high_t_min) / total_timesteps
    expected_weight = (1.0 - high_fraction) + high_fraction * high_t_weight
    return weights / expected_weight


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


def _build_arm(checkpoint: dict, *, width: int, timesteps: int, lr: float,
               device: torch.device) -> tuple[PilotUNet, PilotUNet, torch.optim.Optimizer]:
    model = PilotUNet(width, timesteps).to(device)
    model.load_state_dict({key: value.to(device) for key, value in checkpoint["model"].items()})
    ema = copy.deepcopy(model).eval()
    ema.load_state_dict({key: value.to(device) for key, value in checkpoint["ema"].items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if isinstance(checkpoint.get("optimizer"), dict):
        try:
            optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid checkpoint optimizer state: {exc}") from exc
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        for group in optimizer.param_groups:
            group["lr"] = lr
    model.train()
    return model, ema, optimizer


def _arm_checkpoint(*, config: dict, definitions: dict, model: PilotUNet,
                    ema: PilotUNet, report: dict, arm: str) -> dict:
    return {
        "schema_version": 0,
        "config": copy.deepcopy(config),
        "rules": copy.deepcopy(definitions),
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "ema": {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()},
        "report": copy.deepcopy(report),
        "diagnostic": {
            "kind": "high_noise_finetune_arm",
            "arm": arm,
            "resumable": False,
            "note": "Paired high-noise fine-tuning diagnostic; not resumable.",
        },
    }


def _train_arm(*, model: PilotUNet, ema: PilotUNet, optimizer: torch.optim.Optimizer,
               clean: torch.Tensor, tokens: torch.Tensor, alpha_bar: torch.Tensor,
               steps: int, batch_size: int, dropout: float, ema_decay: float,
               high_t_min: int, high_t_weight: float,
               time_generator: torch.Generator, noise_generator: torch.Generator,
               drop_generator: torch.Generator, device: torch.device) -> dict:
    """Run one arm; the paired arm receives identical generator seeds."""
    total_timesteps = alpha_bar.numel()
    time_counts = [0] * total_timesteps
    weighted_losses = []
    unweighted_losses = []
    start = time.perf_counter()
    for _ in range(steps):
        t = torch.randint(total_timesteps, (batch_size,), device=device, generator=time_generator)
        noise = torch.randn(clean.shape, device=device, generator=noise_generator)
        drop = torch.rand(batch_size, device=device, generator=drop_generator) < dropout
        abar = alpha_bar[t, None, None, None]
        noisy = abar.sqrt() * clean + (1.0 - abar).sqrt() * noise
        prediction = model(noisy, t, tokens, drop)
        per_sample = F.mse_loss(prediction, noise, reduction="none").mean(dim=(1, 2, 3))
        weights = loss_weights(t, high_t_min=high_t_min, high_t_weight=high_t_weight,
                               total_timesteps=total_timesteps)
        loss = (per_sample * weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite fine-tuning loss; no arm checkpoint written")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                average.lerp_(parameter, 1.0 - ema_decay)
        weighted_losses.append(float(loss.detach().cpu()))
        unweighted_losses.append(float(per_sample.mean().detach().cpu()))
        for value in t.detach().cpu().tolist():
            time_counts[value] += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    high_count = sum(time_counts[high_t_min:]) if high_t_min < total_timesteps else 0
    report = {
        "steps": steps,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "last_weighted_loss": weighted_losses[-1],
        "last_unweighted_mse": unweighted_losses[-1],
        "weighted_loss_history": weighted_losses,
        "unweighted_mse_history": unweighted_losses,
        "timestep_counts": {str(index): count for index, count in enumerate(time_counts) if count},
        "high_t_fraction_realized": high_count / (steps * batch_size),
        "high_t_min": high_t_min,
        "high_t_weight": high_t_weight,
        "dropout": dropout,
        "ema_decay": ema_decay,
    }
    return report


def _generators(device: torch.device, seed: int) -> tuple[torch.Generator, torch.Generator, torch.Generator]:
    return (
        torch.Generator(device=device).manual_seed(seed),
        torch.Generator(device=device).manual_seed(seed + 7919),
        torch.Generator(device=device).manual_seed(seed + 104729),
    )


def _metrics_for(checkpoint: dict, target: Any, *, rule: str, timesteps: list[int],
                 seeds: list[int]) -> dict:
    return high_t_baseline.diagnose(
        checkpoint, target, timesteps=timesteps, seeds=seeds, rule=rule
    )


def _comparison(base: dict, other: dict, *, name: str) -> dict:
    rows = {}
    for base_t, other_t in zip(base["results"], other["results"], strict=True):
        timestep = base_t["timestep"]
        rows[str(timestep)] = {}
        for predictor in high_t_baseline.PREDICTORS:
            rows[str(timestep)][predictor] = {
                metric: other_t["averages"][predictor][metric] - base_t["averages"][predictor][metric]
                for metric in ("noise_mse", "x0_mse", "x0_clipped_mse")
            }
    return {"name": name, "rows": rows}


def run_experiment(
    checkpoint: dict,
    target: Any,
    *,
    output_dir: Path | str,
    rule: str | None = None,
    steps: int = 500,
    batch_size: int = 2,
    seed: int = 46,
    lr: float | None = None,
    device: str = "cpu",
    deterministic: bool = False,
    high_t_min: int = 900,
    high_t_weight: float = 5.0,
    eval_timesteps: list[int] | None = None,
    eval_seeds: list[int] | None = None,
    checkpoint_path: Path | str | None = None,
    target_path: Path | str | None = None,
) -> dict:
    """Run matched control and high-noise-weighted fine-tuning arms, then evaluate."""
    config, total_timesteps, width = _checkpoint_parts(checkpoint)
    steps = _validate_int("steps", steps, minimum=1)
    batch_size = _validate_int("batch_size", batch_size, minimum=1)
    seed = _validate_int("seed", seed, minimum=0)
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be a boolean")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if type(high_t_min) is not int or high_t_min < 0 or high_t_min >= total_timesteps:
        raise ValueError(f"high_t_min must be an integer in [0, {total_timesteps - 1}]")
    if not math.isfinite(high_t_weight) or high_t_weight <= 1.0:
        raise ValueError("high_t_weight must be finite and greater than one")
    if eval_timesteps is None:
        eval_timesteps = [20, 100, 500, max(0, total_timesteps - 1)]
    if eval_seeds is None:
        eval_seeds = [0]
    if not isinstance(eval_timesteps, list) or not eval_timesteps or len(set(eval_timesteps)) != len(eval_timesteps):
        raise ValueError("eval_timesteps must be a non-empty list of unique integers")
    if any(type(value) is not int or value < 0 or value >= total_timesteps for value in eval_timesteps):
        raise ValueError("eval_timesteps must be unique indices inside the checkpoint schedule")
    if not isinstance(eval_seeds, list) or not eval_seeds or len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("eval_seeds must be a non-empty list of unique integers")
    if any(type(value) is not int or value < 0 for value in eval_seeds):
        raise ValueError("eval_seeds must contain nonnegative integers")

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output directory already exists; select a new path")

    lr = config.get("lr", 1e-4) if lr is None else lr
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a finite positive number")
    dropout = config.get("dropout", 0.0)
    ema_decay = config.get("ema_decay", 0.999)
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ValueError("checkpoint dropout must be in [0, 1)")
    if not isinstance(ema_decay, (int, float)) or isinstance(ema_decay, bool) or not 0 <= ema_decay < 1:
        raise ValueError("checkpoint EMA decay must be in [0, 1)")

    pilot_train._configure_determinism(deterministic, device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access it")
    target_device = torch.device(device)
    clean, _ = load_target(target)
    clean = clean.to(target_device).repeat(batch_size, 1, 1, 1)
    definitions = checkpoint["rules"]
    rules = load_rules(definitions)
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
    tokens = encode_rules(rules)[names.index(rule):names.index(rule) + 1].to(target_device)
    tokens = tokens.repeat(batch_size, 1, 1)
    _, alpha_bar = schedule(total_timesteps, target_device)

    start_metrics = _metrics_for(checkpoint, target, rule=rule, timesteps=eval_timesteps, seeds=eval_seeds)

    arms = {}
    for arm_name, weight in (("baseline", 1.0), ("weighted", high_t_weight)):
        model, ema, optimizer = _build_arm(
            checkpoint, width=width, timesteps=total_timesteps, lr=float(lr),
            device=target_device,
        )
        time_generator, noise_generator, drop_generator = _generators(target_device, seed)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
            torch.cuda.reset_peak_memory_stats(target_device)
        arm_report = _train_arm(
            model=model, ema=ema, optimizer=optimizer, clean=clean, tokens=tokens,
            alpha_bar=alpha_bar, steps=steps, batch_size=batch_size,
            dropout=float(dropout), ema_decay=float(ema_decay), high_t_min=high_t_min,
            high_t_weight=weight, time_generator=time_generator,
            noise_generator=noise_generator, drop_generator=drop_generator,
            device=target_device,
        )
        arm_report["peak_cuda_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else None
        )
        arm_report["peak_cuda_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(target_device) if target_device.type == "cuda" else None
        )
        arm_config = copy.deepcopy(config)
        arm_config.update({
            "steps": int(config.get("steps", 0)) + steps,
            "batch_size": batch_size,
            "seed": seed,
            "width": width,
            "timesteps": total_timesteps,
            "lr": float(lr),
            "dropout": float(dropout),
            "ema_decay": float(ema_decay),
            "deterministic": deterministic,
            "high_t_min": high_t_min,
            "high_t_weight": weight,
        })
        arm_checkpoint = _arm_checkpoint(
            config=arm_config, definitions=definitions, model=model, ema=ema,
            report=arm_report, arm=arm_name,
        )
        arm_metrics = _metrics_for(arm_checkpoint, target, rule=rule, timesteps=eval_timesteps, seeds=eval_seeds)
        arms[arm_name] = {
            "config": arm_config,
            "training": arm_report,
            "metrics": arm_metrics,
            "checkpoint": arm_checkpoint,
        }

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    output_dir.mkdir(parents=True, exist_ok=False)
    pilot_train.save_checkpoint(output_dir / "baseline.pt", arms["baseline"]["checkpoint"])
    pilot_train.save_checkpoint(output_dir / "weighted.pt", arms["weighted"]["checkpoint"])

    baseline_metrics = arms["baseline"]["metrics"]
    weighted_metrics = arms["weighted"]["metrics"]
    high_fraction = (total_timesteps - high_t_min) / total_timesteps
    expected_weight = (1.0 - high_fraction) + high_fraction * high_t_weight
    report = {
        "schema_version": 1,
        "experiment": "paired_high_noise_finetune",
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "target": str(target_path) if target_path is not None else None,
        "rule": rule,
        "schedule_timesteps": total_timesteps,
        "device": device,
        "deterministic": deterministic,
        "matched": {
            "start_checkpoint": True,
            "optimizer_state": "loaded" if isinstance(checkpoint.get("optimizer"), dict) else "new",
            "target": True,
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "noise_and_dropout_masks": True,
            "full_schedule": total_timesteps,
        },
        "config": {
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "lr": float(lr),
            "dropout": float(dropout),
            "ema_decay": float(ema_decay),
            "high_t_min": high_t_min,
            "high_t_weight": high_t_weight,
            "eval_timesteps": eval_timesteps,
            "eval_seeds": eval_seeds,
        },
        "objective_weights": {
            "high_t_fraction_under_uniform_t": high_fraction,
            "expected_weight": expected_weight,
            "normalized_low_t_weight": 1.0 / expected_weight,
            "normalized_high_t_weight": high_t_weight / expected_weight,
        },
        "start_metrics": start_metrics,
        "arms": {
            "baseline": {
                "high_t_weight": 1.0,
                "config": arms["baseline"]["config"],
                "training": arms["baseline"]["training"],
                "metrics": baseline_metrics,
            },
            "weighted": {
                "high_t_weight": high_t_weight,
                "config": arms["weighted"]["config"],
                "training": arms["weighted"]["training"],
                "metrics": weighted_metrics,
            },
        },
        "comparison": {
            "baseline_minus_start": _comparison(start_metrics, baseline_metrics, name="baseline_minus_start"),
            "weighted_minus_start": _comparison(start_metrics, weighted_metrics, name="weighted_minus_start"),
            "weighted_minus_baseline": _comparison(baseline_metrics, weighted_metrics, name="weighted_minus_baseline"),
        },
        "note": (
            "Paired fixed-scene fine-tuning diagnostic. Both arms start from the same checkpoint and see "
            "the same timesteps, noise, and dropout masks; only the normalized high-timestep loss weight "
            "differs. Full diffusion schedule retained. This is not a population result, human audit, "
            "Stage 0 gate, or checkpoint-compatibility change."
        ),
    }
    report_path = output_dir / "report.json"
    report["outputs"] = {
        "baseline_checkpoint": str(output_dir / "baseline.pt"),
        "weighted_checkpoint": str(output_dir / "weighted.pt"),
        "report": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired high-noise fixed-scene fine-tuning diagnostic")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwritten")
    parser.add_argument("--rule", type=str, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--high-t-min", type=int, default=900)
    parser.add_argument("--high-t-weight", type=float, default=5.0)
    parser.add_argument("--eval-timesteps", type=int, nargs="+", default=[20, 100, 500, 900, 990, 999])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    try:
        if not args.checkpoint.is_file():
            raise ValueError(f"Checkpoint file not found: {args.checkpoint}")
        if not args.target.is_file():
            raise ValueError(f"Target image file not found: {args.target}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        target_hash = hashlib.sha256(args.target.read_bytes()).hexdigest()
        report = run_experiment(
            checkpoint, args.target, output_dir=args.output_dir, rule=args.rule,
            steps=args.steps, batch_size=args.batch_size, seed=args.seed, lr=args.lr,
            device=args.device, deterministic=args.deterministic, high_t_min=args.high_t_min,
            high_t_weight=args.high_t_weight, eval_timesteps=args.eval_timesteps,
            eval_seeds=args.eval_seeds, checkpoint_path=args.checkpoint,
            target_path=args.target,
        )
        report["checkpoint_sha256"] = checkpoint_hash
        report["target_sha256"] = target_hash
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
