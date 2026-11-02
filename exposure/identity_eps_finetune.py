"""Paired fixed-scene experiment: standard versus identity-residual noise prediction.

Both arms use the same ``PilotUNet`` backbone, initialization, fixed target,
timesteps, noise draws, dropout masks, optimizer settings, and EMA update. The
control predicts noise directly. The treatment predicts ``x + f(x, t, c)`` so
the high-noise identity solution is available without passing through the
network's downsampling and output head.

The full diffusion schedule is retained. This is an isolated diagnostic, not a
population result or a Stage 0 gate. Saved checkpoints are schema version 0 and
non-resumable; identity-residual checkpoints require the sampler in this module.
The report also includes a constant-target-mean image baseline and a pure
identity-epsilon sampler (the treatment path with a zero residual) so sample
MSE can be interpreted against trivial controls.
"""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from exposure import pilot_train
from exposure.denoise_probe import reconstruct_x0
from exposure.measure import unique_object
from exposure.pilot_train import PilotUNet, encode_rules, load_rules, schedule
from exposure.reconstruction_probe import load_target


MODES = ("standard", "identity_residual")


class _ZeroBackbone(torch.nn.Module):
    """The identity-residual backbone that always returns zero."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor,
                drop: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros_like(x)


def _validate_int(name: str, value: object, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


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


def predict_noise(backbone: torch.nn.Module, noisy: torch.Tensor, timestep: torch.Tensor,
                  tokens: torch.Tensor, drop: torch.Tensor | None, mode: str) -> torch.Tensor:
    """Apply one output parameterization to the same backbone."""
    if mode not in MODES:
        raise ValueError(f"Unknown predictor mode {mode!r}")
    base = backbone(noisy, timestep, tokens, drop)
    if mode == "standard":
        return base
    return noisy + base


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_model(width: int, timesteps: int, device: torch.device, init_seed: int) -> PilotUNet:
    torch.manual_seed(init_seed)
    model = PilotUNet(width, timesteps)
    # Both arms get the same neutral head. The control then starts at eps=0;
    # the identity-residual arm starts at eps=x, the copy predictor.
    final = model.output[-1]
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)
    return model.to(device)


@torch.no_grad()
def _one_step_metrics(model: torch.nn.Module, mode: str, *, clean: torch.Tensor,
                      tokens: torch.Tensor, alpha_bar: torch.Tensor, timesteps: list[int],
                      seeds: list[int], device: torch.device) -> dict:
    results = []
    for timestep in timesteps:
        alpha = alpha_bar[timestep]
        time = torch.tensor([timestep], device=device)
        seed_results = []
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
            predictions = {
                "model": predict_noise(model, noisy, time, tokens, None, mode),
                "copy": noisy,
                "zero": torch.zeros_like(noise),
                "oracle": noise,
            }
            metrics = {}
            for name, predicted in predictions.items():
                reconstructed = reconstruct_x0(noisy, predicted, alpha)
                clipped = reconstructed.clamp(-1.0, 1.0)
                metrics[name] = {
                    "noise_mse": float((predicted - noise).square().mean()),
                    "x0_mse": float((reconstructed - clean).square().mean()),
                    "x0_clipped_mse": float((clipped - clean).square().mean()),
                }
                if not all(math.isfinite(value) for value in metrics[name].values()):
                    raise ValueError("Non-finite one-step result")
            seed_results.append({"seed": seed, "predictors": metrics})
        averages = {
            name: {
                metric: sum(row["predictors"][name][metric] for row in seed_results) / len(seed_results)
                for metric in ("noise_mse", "x0_mse", "x0_clipped_mse")
            }
            for name in ("model", "copy", "zero", "oracle")
        }
        results.append({"timestep": timestep, "seeds": seed_results, "averages": averages})
    return {"results": results}


def _train_arm(*, model: PilotUNet, ema: PilotUNet, optimizer: torch.optim.Optimizer,
               mode: str, clean: torch.Tensor, tokens: torch.Tensor,
               alpha_bar: torch.Tensor, steps: int, batch_size: int, dropout: float,
               ema_decay: float, time_generator: torch.Generator,
               noise_generator: torch.Generator, drop_generator: torch.Generator,
               device: torch.device) -> dict:
    losses = []
    time_counts = [0] * alpha_bar.numel()
    start = time.perf_counter()
    for _ in range(steps):
        t = torch.randint(alpha_bar.numel(), (batch_size,), device=device, generator=time_generator)
        noise = torch.randn(clean.shape, device=device, generator=noise_generator)
        drop = torch.rand(batch_size, device=device, generator=drop_generator) < dropout
        abar = alpha_bar[t, None, None, None]
        noisy = abar.sqrt() * clean + (1.0 - abar).sqrt() * noise
        prediction = predict_noise(model, noisy, t, tokens, drop, mode)
        loss = F.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; no checkpoint written")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                average.lerp_(parameter, 1.0 - ema_decay)
        losses.append(float(loss.detach().cpu()))
        for value in t.detach().cpu().tolist():
            time_counts[value] += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "mode": mode,
        "steps": steps,
        "batch_size": batch_size,
        "elapsed_seconds": time.perf_counter() - start,
        "last_loss": losses[-1],
        "loss_history": losses,
        "timestep_counts": {str(index): count for index, count in enumerate(time_counts) if count},
        "dropout": dropout,
        "ema_decay": ema_decay,
    }


@torch.no_grad()
def _sample(model: torch.nn.Module, mode: str, *, tokens: torch.Tensor,
            beta: torch.Tensor, alpha_bar: torch.Tensor, seed: int,
            device: torch.device) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    total_timesteps = alpha_bar.numel()
    x = torch.randn((1, 3, 64, 64), device=device, generator=generator)
    for step in range(total_timesteps - 1, -1, -1):
        time = torch.tensor([step], device=device)
        prediction = predict_noise(model, x, time, tokens, None, mode)
        x = (x - beta[step] / (1.0 - alpha_bar[step]).sqrt() * prediction) / (1.0 - beta[step]).sqrt()
        if step:
            variance = beta[step] * (1.0 - alpha_bar[step - 1]) / (1.0 - alpha_bar[step])
            x = x + variance.sqrt() * torch.randn(x.shape, device=device, generator=generator)
    if not torch.isfinite(x).all():
        raise RuntimeError("Non-finite sample trajectory; no output written")
    pixels = ((x[0].clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels, mode="RGB")


def _arm_checkpoint(*, config: dict, definitions: dict, model: PilotUNet, ema: PilotUNet,
                    report: dict, mode: str) -> dict:
    return {
        "schema_version": 0,
        "config": copy.deepcopy(config),
        "rules": copy.deepcopy(definitions),
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "ema": {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()},
        "report": copy.deepcopy(report),
        "diagnostic": {
            "kind": "identity_eps_finetune_arm",
            "mode": mode,
            "resumable": False,
            "requires_custom_sampler": mode == "identity_residual",
            "note": "Fixed-scene paired diagnostic; not resumable.",
        },
    }


def _comparison(base: dict, other: dict, *, name: str) -> dict:
    rows = {}
    for base_t, other_t in zip(base["results"], other["results"], strict=True):
        timestep = base_t["timestep"]
        rows[str(timestep)] = {}
        for predictor in ("model", "copy", "zero", "oracle"):
            rows[str(timestep)][predictor] = {
                metric: other_t["averages"][predictor][metric] - base_t["averages"][predictor][metric]
                for metric in ("noise_mse", "x0_mse", "x0_clipped_mse")
            }
    return {"name": name, "rows": rows}


def run_experiment(
    definitions: dict,
    target: Any,
    *,
    output_dir: Path | str,
    rule: str | None = None,
    steps: int = 4000,
    batch_size: int = 2,
    seed: int = 46,
    init_seed: int = 46,
    width: int = 16,
    timesteps: int = 1000,
    lr: float = 1e-4,
    dropout: float = 0.0,
    ema_decay: float = 0.999,
    device: str = "cpu",
    deterministic: bool = False,
    eval_timesteps: list[int] | None = None,
    eval_seeds: list[int] | None = None,
    sample_seeds: list[int] | None = None,
    target_path: Path | str | None = None,
) -> dict:
    """Train matched standard and identity-residual arms from scratch."""
    steps = _validate_int("steps", steps, minimum=1)
    batch_size = _validate_int("batch_size", batch_size, minimum=1)
    seed = _validate_int("seed", seed, minimum=0)
    init_seed = _validate_int("init_seed", init_seed, minimum=0)
    width = _validate_int("width", width, minimum=4)
    timesteps = _validate_int("timesteps", timesteps, minimum=2)
    if width % 4:
        raise ValueError("width must be a multiple of four")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a finite positive number")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not isinstance(ema_decay, (int, float)) or isinstance(ema_decay, bool) or not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be a boolean")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if eval_timesteps is None:
        eval_timesteps = [20, 100, 500, 900, 990, timesteps - 1]
    if eval_seeds is None:
        eval_seeds = [0]
    if sample_seeds is None:
        sample_seeds = [0]
    eval_timesteps = _unique_ints("eval_timesteps", eval_timesteps, minimum=0, maximum=timesteps - 1)
    eval_seeds = _unique_ints("eval_seeds", eval_seeds, minimum=0)
    sample_seeds = _unique_ints("sample_seeds", sample_seeds, minimum=0)

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Output directory already exists; select a new path")

    rules = load_rules(definitions)
    if not rules:
        raise ValueError("definitions contain no rules")
    if rule is None:
        if len(rules) == 1:
            rule = next(iter(rules))
        else:
            raise ValueError(f"rule must be specified when definitions contain multiple rules: {list(rules)}")
    elif rule not in rules:
        raise ValueError(f"Unknown rule {rule!r}; available rules: {list(rules)}")

    pilot_train._configure_determinism(deterministic, device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access it")
    target_device = torch.device(device)
    clean, derived_path = load_target(target)
    clean = clean.to(target_device)
    target_display = str(target_path or derived_path) if (target_path or derived_path) else None
    target_pixels = ((clean[0].cpu() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    target_float = target_pixels.astype(np.float32) / 255.0
    constant_pixels = target_float.mean(axis=(0, 1), keepdims=True)
    constant_mse = float(np.mean((constant_pixels - target_float) ** 2))

    names = list(rules)
    tokens = encode_rules(rules)[names.index(rule):names.index(rule) + 1].to(target_device)
    train_clean = clean.repeat(batch_size, 1, 1, 1)
    train_tokens = tokens.repeat(batch_size, 1, 1)
    beta, alpha_bar = schedule(timesteps, target_device)
    terminal_alpha_bar = float(alpha_bar[-1].cpu())
    terminal_amplification = (
        float(((1.0 - alpha_bar[-1]) / alpha_bar[-1]).sqrt().cpu())
        if terminal_alpha_bar > 0 else None
    )

    identity_epsilon_baseline = []
    zero_backbone = _ZeroBackbone().to(target_device)
    for sample_seed in sample_seeds:
        image = _sample(
            zero_backbone, "identity_residual", tokens=tokens, beta=beta,
            alpha_bar=alpha_bar, seed=sample_seed, device=target_device,
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        baseline_mse = float(np.mean((pixels - target_float) ** 2))
        if not math.isfinite(baseline_mse):
            raise RuntimeError("Non-finite identity-epsilon baseline MSE; no output written")
        identity_epsilon_baseline.append({
            "seed": sample_seed,
            "mse_0_1": baseline_mse,
            "image": image,
        })

    baseline = _build_model(width, timesteps, target_device, init_seed)
    identity = _build_model(width, timesteps, target_device, init_seed)
    initial_hashes = {"standard": _model_hash(baseline), "identity_residual": _model_hash(identity)}
    if initial_hashes["standard"] != initial_hashes["identity_residual"]:
        raise RuntimeError("Matched arm initialization failed")

    initial_metrics = {
        mode: _one_step_metrics(model, mode, clean=clean, tokens=tokens,
                                alpha_bar=alpha_bar, timesteps=eval_timesteps,
                                seeds=eval_seeds, device=target_device)
        for mode, model in (("standard", baseline), ("identity_residual", identity))
    }

    arms = {}
    for mode, model in (("standard", baseline), ("identity_residual", identity)):
        ema = copy.deepcopy(model).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
        time_generator, noise_generator, drop_generator = (
            torch.Generator(device=target_device).manual_seed(seed),
            torch.Generator(device=target_device).manual_seed(seed + 7919),
            torch.Generator(device=target_device).manual_seed(seed + 104729),
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
            torch.cuda.reset_peak_memory_stats(target_device)
        training = _train_arm(
            model=model, ema=ema, optimizer=optimizer, mode=mode, clean=train_clean,
            tokens=train_tokens, alpha_bar=alpha_bar, steps=steps, batch_size=batch_size,
            dropout=float(dropout), ema_decay=float(ema_decay),
            time_generator=time_generator, noise_generator=noise_generator,
            drop_generator=drop_generator, device=target_device,
        )
        training["peak_cuda_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else None
        )
        training["peak_cuda_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(target_device) if target_device.type == "cuda" else None
        )
        metrics = {
            "raw": _one_step_metrics(
                model, mode, clean=clean, tokens=tokens, alpha_bar=alpha_bar,
                timesteps=eval_timesteps, seeds=eval_seeds, device=target_device,
            ),
            "ema": _one_step_metrics(
                ema, mode, clean=clean, tokens=tokens, alpha_bar=alpha_bar,
                timesteps=eval_timesteps, seeds=eval_seeds, device=target_device,
            ),
        }
        samples = []
        for sample_seed in sample_seeds:
            image = _sample(ema, mode, tokens=tokens, beta=beta, alpha_bar=alpha_bar,
                            seed=sample_seed, device=target_device)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            sample_mse = float(np.mean((pixels - target_float) ** 2))
            if not math.isfinite(sample_mse):
                raise RuntimeError("Non-finite sample image MSE; no output written")
            samples.append({
                "seed": sample_seed,
                "mse_0_1": sample_mse,
                "image": image,
            })
        arm_config = {
            "mode": mode,
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "init_seed": init_seed,
            "width": width,
            "timesteps": timesteps,
            "lr": float(lr),
            "dropout": float(dropout),
            "ema_decay": float(ema_decay),
            "deterministic": deterministic,
        }
        checkpoint = _arm_checkpoint(
            config=arm_config, definitions=definitions, model=model, ema=ema,
            report=training, mode=mode,
        )
        arms[mode] = {
            "config": arm_config,
            "training": training,
            "metrics": metrics,
            "samples": samples,
            "checkpoint": checkpoint,
        }

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    output_dir.mkdir(parents=True, exist_ok=False)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir()
    sample_paths = {}
    for mode in MODES:
        pilot_train.save_checkpoint(output_dir / f"{mode}.pt", arms[mode]["checkpoint"])
        sample_paths[mode] = []
        for sample in arms[mode]["samples"]:
            path = sample_dir / f"{mode}-{sample['seed']}.png"
            sample["image"].save(path)
            sample_paths[mode].append({"seed": sample["seed"], "path": str(path), "mse_0_1": sample["mse_0_1"]})
            del sample["image"]

    identity_epsilon_paths = []
    for sample in identity_epsilon_baseline:
        path = sample_dir / f"identity-epsilon-{sample['seed']}.png"
        sample["image"].save(path)
        identity_epsilon_paths.append({
            "seed": sample["seed"],
            "path": str(path),
            "mse_0_1": sample["mse_0_1"],
        })
        del sample["image"]

    report = {
        "schema_version": 1,
        "experiment": "paired_fixed_scene_identity_eps",
        "target": target_display,
        "rule": rule,
        "schedule_timesteps": timesteps,
        "terminal_alpha_bar": terminal_alpha_bar,
        "terminal_epsilon_to_x0_amplification": terminal_amplification,
        "prior_approximation_warning": terminal_alpha_bar > 0.01,
        "device": device,
        "deterministic": deterministic,
        "matched": {
            "initial_parameter_sha256": initial_hashes,
            "same_initialization": initial_hashes["standard"] == initial_hashes["identity_residual"],
            "same_target": True,
            "same_realized_timesteps": (
                arms["standard"]["training"]["timestep_counts"]
                == arms["identity_residual"]["training"]["timestep_counts"]
            ),
            "shared_draw_streams": True,
            "generator_seeds": {
                "time": seed,
                "noise": seed + 7919,
                "dropout": seed + 104729,
            },
            "same_optimizer_and_ema_settings": True,
            "full_schedule": timesteps,
        },
        "config": {
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "init_seed": init_seed,
            "width": width,
            "timesteps": timesteps,
            "lr": float(lr),
            "dropout": float(dropout),
            "ema_decay": float(ema_decay),
            "eval_timesteps": eval_timesteps,
            "eval_seeds": eval_seeds,
            "sample_seeds": sample_seeds,
        },
        "initial_metrics": initial_metrics,
        "baselines": {
            "constant_target_mean_mse_0_1": constant_mse,
            "target_mean_rgb_0_1": [float(value) for value in constant_pixels.reshape(3)],
            "identity_epsilon_sampler": identity_epsilon_paths,
        },
        "arms": {
            "standard": {
                "mode": "standard",
                "config": arms["standard"]["config"],
                "training": arms["standard"]["training"],
                "metrics": arms["standard"]["metrics"],
                "samples": sample_paths["standard"],
            },
            "identity_residual": {
                "mode": "identity_residual",
                "config": arms["identity_residual"]["config"],
                "training": arms["identity_residual"]["training"],
                "metrics": arms["identity_residual"]["metrics"],
                "samples": sample_paths["identity_residual"],
            },
        },
        "comparison": {
            "raw_identity_minus_standard": _comparison(
                arms["standard"]["metrics"]["raw"], arms["identity_residual"]["metrics"]["raw"],
                name="raw_identity_minus_standard",
            ),
            "ema_identity_minus_standard": _comparison(
                arms["standard"]["metrics"]["ema"], arms["identity_residual"]["metrics"]["ema"],
                name="ema_identity_minus_standard",
            ),
        },
        "outputs": {
            "standard_checkpoint": str(output_dir / "standard.pt"),
            "identity_checkpoint": str(output_dir / "identity_residual.pt"),
            "samples": str(sample_dir),
            "report": str(output_dir / "report.json"),
        },
        "note": (
            "Paired fixed-scene diagnostic comparing standard noise prediction with "
            "eps_hat = x + f(x,t,c). Both arms use the same initialization, target, "
            "timesteps, noise, dropout, optimizer, and EMA settings. Full schedule retained. "
            "Sample MSE should be read against the constant-target-mean baseline; per-pixel MSE "
            "does not measure rule satisfaction. This is not a population result, human audit, "
            "or Stage 0 gate."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired fixed-scene standard versus identity-residual experiment")
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwritten")
    parser.add_argument("--rule", type=str, default=None)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--init-seed", type=int, default=46)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-timesteps", type=int, nargs="+", default=[20, 100, 500, 900, 990, 999])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    try:
        if not args.rules.is_file():
            raise ValueError(f"Rules file not found: {args.rules}")
        if not args.target.is_file():
            raise ValueError(f"Target image file not found: {args.target}")
        definitions = json.loads(args.rules.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        rules_hash = hashlib.sha256(args.rules.read_bytes()).hexdigest()
        target_hash = hashlib.sha256(args.target.read_bytes()).hexdigest()
        report = run_experiment(
            definitions, args.target, output_dir=args.output_dir, rule=args.rule,
            steps=args.steps, batch_size=args.batch_size, seed=args.seed,
            init_seed=args.init_seed, width=args.width, timesteps=args.timesteps,
            lr=args.lr, dropout=args.dropout, ema_decay=args.ema_decay,
            device=args.device, deterministic=args.deterministic,
            eval_timesteps=args.eval_timesteps, eval_seeds=args.eval_seeds,
            sample_seeds=args.sample_seeds, target_path=args.target,
        )
        report["rules"] = str(args.rules)
        report["rules_sha256"] = rules_hash
        report["target_sha256"] = target_hash
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
