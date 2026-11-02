"""Fixed-scene, paired comparison of learned and sinusoidal timestep embeddings.

Experimental checkpoints use distinct time parameterizations and are not
compatible with the regular pilot sampler. This does not assess Stage 0.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from exposure import pilot_train
from exposure.identity_eps_finetune import _one_step_metrics, _sample, _train_arm
from exposure.measure import unique_object
from exposure.reconstruction_probe import load_target


class SinusoidalTime(torch.nn.Module):
    """Deterministic transformer-style embedding of discrete diffusion time."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("frequencies", torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2)))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = timestep.float().unsqueeze(-1) * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


def build_models(width: int, timesteps: int, init_seed: int, device: torch.device):
    torch.manual_seed(init_seed)
    learned = pilot_train.PilotUNet(width, timesteps)
    sinusoidal = copy.deepcopy(learned)
    sinusoidal.time = SinusoidalTime(width * 4)
    # The non-time parameters start byte-for-byte equal. Time representations
    # and parameter counts differ by design.
    return learned.to(device), sinusoidal.to(device)


def run(definitions: dict, target: Path, output: Path, *, rule: str, steps: int,
        batch_size: int, seed: int, init_seed: int, width: int, timesteps: int,
        lr: float, device: str, deterministic: bool, eval_times: list[int],
        eval_seeds: list[int], sample_seeds: list[int]) -> dict:
    if output.exists():
        raise ValueError("Output directory already exists")
    if steps < 1 or batch_size < 1 or min(seed, init_seed) < 0 or timesteps < 2:
        raise ValueError("Invalid training configuration")
    if not math.isfinite(lr) or lr <= 0 or device not in ("cpu", "cuda"):
        raise ValueError("Invalid learning rate or device")
    for name, values, upper in (("eval times", eval_times, timesteps),
                                ("eval seeds", eval_seeds, None),
                                ("sample seeds", sample_seeds, None)):
        if not values or len(set(values)) != len(values) or any(type(v) is not int or v < 0 or (upper is not None and v >= upper) for v in values):
            raise ValueError(f"Invalid {name}")
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
    models = build_models(width, timesteps, init_seed, dev)
    arms = {}
    target_np = ((clean[0].cpu().permute(1, 2, 0).numpy() + 1) / 2)
    output.mkdir(parents=True, exist_ok=False)
    for name, model in zip(("learned", "sinusoidal"), models, strict=True):
        ema = copy.deepcopy(model).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        generators = [torch.Generator(device=dev).manual_seed(seed + offset) for offset in (0, 7919, 104729)]
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        training = _train_arm(model=model, ema=ema, optimizer=optimizer, mode="standard",
            clean=clean.repeat(batch_size, 1, 1, 1), tokens=tokens.repeat(batch_size, 1, 1),
            alpha_bar=alpha_bar, steps=steps, batch_size=batch_size, dropout=0.0, ema_decay=0.999,
            time_generator=generators[0], noise_generator=generators[1], drop_generator=generators[2], device=dev)
        training["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
        metrics = {label: _one_step_metrics(net, "standard", clean=clean, tokens=tokens,
                   alpha_bar=alpha_bar, timesteps=eval_times, seeds=eval_seeds, device=dev)
                   for label, net in (("raw", model), ("ema", ema))}
        samples = []
        for sample_seed in sample_seeds:
            image = _sample(ema, "standard", tokens=tokens, beta=beta, alpha_bar=alpha_bar, seed=sample_seed, device=dev)
            path = output / f"{name}-{sample_seed}.png"
            image.save(path)
            pixels = np.asarray(image, dtype=np.float32) / 255
            samples.append({"seed": sample_seed, "path": str(path), "mse_0_1": float(np.mean((pixels - target_np) ** 2))})
        checkpoint = {"schema_version": 0, "config": {"width": width, "timesteps": timesteps},
                      "rules": definitions, "model": model.state_dict(), "ema": ema.state_dict(),
                      "diagnostic": {"kind": "time_embedding_probe", "time_embedding": name,
                                     "requires_custom_sampler": name == "sinusoidal", "resumable": False}}
        pilot_train.save_checkpoint(output / f"{name}.pt", checkpoint)
        arms[name] = {"training": training, "metrics": metrics, "samples": samples,
                      "parameter_count": sum(p.numel() for p in model.parameters())}
    report = {"schema_version": 1, "rule": rule, "target": str(target),
              "config": {"steps": steps, "batch_size": batch_size, "seed": seed, "init_seed": init_seed,
                         "width": width, "timesteps": timesteps, "lr": lr, "device": device,
                         "deterministic": deterministic, "dropout": 0.0, "ema_decay": 0.999,
                         "eval_times": eval_times, "eval_seeds": eval_seeds, "sample_seeds": sample_seeds},
              "arms": arms, "constant_target_mean_mse_0_1": float(np.mean((target_np - target_np.mean(axis=(0, 1))) ** 2)),
              "note": "Single fixed scene and paired random streams; no Stage 0 or generalization claim. Sinusoidal checkpoint requires the custom time architecture."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--init-seed", type=int, default=46)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-times", type=int, nargs="+", default=[20, 100, 500, 900, 999])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=[5, 6])
    args = parser.parse_args()
    try:
        definitions = json.loads(args.rules.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        report = run(definitions, args.target, args.output, rule=args.rule, steps=args.steps,
                     batch_size=args.batch_size, seed=args.seed, init_seed=args.init_seed, width=args.width,
                     timesteps=args.timesteps, lr=args.lr, device=args.device, deterministic=args.deterministic,
                     eval_times=args.eval_times, eval_seeds=args.eval_seeds, sample_seeds=args.sample_seeds)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
