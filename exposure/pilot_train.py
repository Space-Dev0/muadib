"""Small online conditional diffusion instrument, not a research result.

Uses predicate/object-slot tokens, cross-attention, DDPM noise prediction,
classifier-free dropout, AdamW and EMA. Version 3 checkpoints can resume on
the same device type and compatible runtime, preserving random streams. Resume
checks Python/PyTorch/NumPy/Pillow versions and fingerprints project code used
by the pilot; exact results across hardware or drivers are not guaranteed.
Sampling and loss are preliminary; there is no validation, guidance sweep, or
Stage 0 gate decision. Strict deterministic training is opt-in and changes
process-wide PyTorch backend flags; unsupported operations raise errors. New
checkpoints record that mode and sampling uses it too. Older v1/v2 checkpoints
can be sampled in the default non-strict mode, but cannot be resumed.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from exposure import measure as _measure_module, render as _render_module, rules as _rules_module
from exposure.measure import load_rules, unique_object
from exposure.render import COLORS, SHAPES, render_conditioned
from exposure.rules import Attribute, satisfies


# A CUDA context already initialized by someone else cannot be proven to have
# seen the cuBLAS workspace setting. Remember only configurations established
# by this module before CUDA initialization in the current process.
_CUBLAS_CONFIGURED_BEFORE_INIT: str | None = None


def _configure_determinism(enabled: bool, device: str) -> None:
    """Configure process-wide PyTorch flags; strict mode may reject unsupported ops."""
    global _CUBLAS_CONFIGURED_BEFORE_INIT
    if type(enabled) is not bool:
        raise ValueError("deterministic must be a boolean")
    if enabled and device == "cuda":
        value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if value is not None and value not in {":4096:8", ":16:8"}:
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA")
        if torch.cuda.is_initialized():
            if value is None or value != _CUBLAS_CONFIGURED_BEFORE_INIT:
                raise ValueError("CUDA was initialized before deterministic setup; start a new process")
        else:
            value = value or ":4096:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
            _CUBLAS_CONFIGURED_BEFORE_INIT = value
    torch.use_deterministic_algorithms(enabled, warn_only=False)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def encode_rules(rules: dict) -> torch.Tensor:
    """Encode predicate kind, subject slot, object slot and attribute value."""
    max_predicates = max(len(rule) for rule in rules.values())
    tokens = torch.zeros((len(rules), max_predicates, 4), dtype=torch.long)
    values = {value: index + 1 for index, value in enumerate((*COLORS, *SHAPES))}
    for row, predicates in enumerate(rules.values()):
        variables = sorted({variable for p in predicates for variable in
                            ((p.variable,) if isinstance(p, Attribute) else (p.subject, p.object))})
        if len(variables) > 3:
            raise ValueError("The pilot supports at most three object variables per rule")
        slots = {variable: i + 1 for i, variable in enumerate(variables)}
        for col, predicate in enumerate(predicates):
            if isinstance(predicate, Attribute):
                tokens[row, col] = torch.tensor([1 if predicate.field == "color" else 2,
                                                 slots[predicate.variable], 0, values[predicate.value]])
            else:
                tokens[row, col] = torch.tensor([3 if predicate.kind == "left_of" else 4,
                                                 slots[predicate.subject], slots[predicate.object], 0])
    return tokens


class Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time_embedding))[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class PilotUNet(nn.Module):
    """A compact 64x64 U-Net with one cross-attention block at 16x16."""
    def __init__(self, width: int, timesteps: int):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("width must be a positive multiple of four, at least four")
        dim = width * 4
        self.time = nn.Embedding(timesteps, dim)
        self.kind = nn.Embedding(5, dim, padding_idx=0)
        self.subject = nn.Embedding(4, dim, padding_idx=0)
        self.object = nn.Embedding(4, dim, padding_idx=0)
        self.value = nn.Embedding(8, dim, padding_idx=0)
        self.null = nn.Parameter(torch.zeros(dim))
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.enc1 = Residual(width, width, dim)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.enc2 = Residual(width * 2, width * 2, dim)
        self.down2 = nn.Conv2d(width * 2, dim, 3, stride=2, padding=1)
        self.middle = Residual(dim, dim, dim)
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.dec2 = Residual(dim + width * 2, width * 2, dim)
        self.dec1 = Residual(width * 3, width, dim)
        self.output = nn.Sequential(nn.GroupNorm(4, width), nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor,
                drop: torch.Tensor | None = None) -> torch.Tensor:
        mask = tokens[:, :, 0] != 0
        context = (self.kind(tokens[:, :, 0]) + self.subject(tokens[:, :, 1]) +
                   self.object(tokens[:, :, 2]) + self.value(tokens[:, :, 3]))
        if drop is not None:
            mask = mask & ~drop[:, None]
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            # Always expose one null token so attention never sees all padding.
            context = context.clone()
            mask = mask.clone()
            context[empty, 0] = self.null
            mask[empty, 0] = True
        emb = self.time(t)
        first = self.enc1(self.input(x), emb)
        second = self.enc2(self.down1(first), emb)
        middle = self.middle(self.down2(second), emb)
        sequence = middle.flatten(2).transpose(1, 2)
        attended, _ = self.attention(sequence, context, context, key_padding_mask=~mask, need_weights=False)
        middle = middle + attended.transpose(1, 2).reshape_as(middle)
        up = F.interpolate(middle, size=second.shape[-2:], mode="nearest")
        up = self.dec2(torch.cat((up, second), dim=1), emb)
        up = F.interpolate(up, size=first.shape[-2:], mode="nearest")
        return self.output(self.dec1(torch.cat((up, first), dim=1), emb))


def schedule(timesteps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if timesteps < 2:
        raise ValueError("timesteps must be at least two")
    beta = torch.linspace(1e-4, 0.02, timesteps, device=device)
    return beta, torch.cumprod(1.0 - beta, dim=0)


def require_standard_predictor(checkpoint: object) -> None:
    """Reject explicitly marked checkpoints requiring another predictor."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    diagnostic = checkpoint.get("diagnostic")
    if isinstance(diagnostic, dict) and diagnostic.get("requires_custom_sampler") is True:
        raise ValueError("Checkpoint requires a custom sampler/predictor; standard pilot tools cannot use it")


def _weights(names: list[str], raw: object | None) -> list[float]:
    if raw is None:
        return [1.0 / len(names)] * len(names)
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise ValueError("Weights must name each and only each rule")
    weights = [raw[name] for name in names]
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
           for w in weights) or not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Weights must be finite, nonnegative, and sum to one")
    return [float(w) for w in weights]


def _runtime(device: str) -> dict:
    """Identify RNG implementations and, for CUDA, the visible device layout."""
    paths = {"pilot_train": Path(__file__), "measure": Path(_measure_module.__file__),
             "render": Path(_render_module.__file__), "rules": Path(_rules_module.__file__)}
    fingerprints = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    info = {"python": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
            "torch": str(torch.__version__), "device": device, "cuda_version": None,
            "numpy": np.__version__, "pillow": PIL.__version__, "code_sha256": fingerprints,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None,
            "cuda_devices": []}
    if device == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["cuda_devices"] = [
            {"name": torch.cuda.get_device_name(i),
             "capability": list(torch.cuda.get_device_capability(i))}
            for i in range(torch.cuda.device_count())
        ]
    return info


def _resume_parts(checkpoint: object) -> tuple[dict, dict, dict]:
    """Validate presence of state needed to resume a version 3 checkpoint."""
    required = {"schema_version", "config", "rules", "model", "ema", "optimizer", "report", "rng"}
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 3 or not required <= checkpoint.keys():
        raise ValueError("Checkpoint lacks required resume state; version 3 is required")
    config, report, rng = checkpoint["config"], checkpoint["report"], checkpoint["rng"]
    config_keys = {"steps", "batch_size", "seed", "width", "timesteps", "lr", "dropout", "ema_decay", "max_attempts", "deterministic"}
    report_keys = {"device", "weights", "condition_counts", "realized_rule_counts_by_condition",
                   "sampling_attempts", "elapsed_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"}
    if (not isinstance(config, dict) or not config_keys <= config.keys() or
            not isinstance(report, dict) or not report_keys <= report.keys() or
            not isinstance(rng, dict) or set(rng) != {"runtime", "python", "torch_cpu", "torch_cuda"}):
        raise ValueError("Checkpoint lacks required resume state")
    if type(config["steps"]) is not int or config["steps"] < 1:
        raise ValueError("Checkpoint has invalid completed step count")
    return config, report, rng


def train(definitions: dict, weights: object | None, *, steps: int, batch_size: int, seed: int,
          device: str, width: int, timesteps: int, lr: float, dropout: float,
          ema_decay: float, max_attempts: int, deterministic: bool = False,
          resume: dict | None = None) -> tuple[dict, dict]:
    """Run updates, optionally restoring a v3 checkpoint; changes global torch flags."""
    if steps < 1 or batch_size < 1 or seed < 0 or max_attempts < 1:
        raise ValueError("steps, batch size, and attempts must be positive; seed must be nonnegative")
    if not math.isfinite(lr) or lr <= 0 or not 0 <= dropout < 1 or not 0 <= ema_decay < 1:
        raise ValueError("invalid learning rate, dropout or EMA decay")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    _configure_determinism(deterministic, device)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access it")
    rules = load_rules(definitions)
    names = list(rules)
    probs = _weights(names, weights)
    tokens = encode_rules(rules).to(target)
    runtime = _runtime(device)
    previous_steps, previous_elapsed = 0, 0.0
    previous_allocated, previous_reserved = None, None
    prior_report = prior_rng = None
    if resume is not None:
        prior_config, prior_report, prior_rng = _resume_parts(resume)
        expected = {"batch_size": batch_size, "seed": seed, "width": width, "timesteps": timesteps,
                    "lr": lr, "dropout": dropout, "ema_decay": ema_decay, "max_attempts": max_attempts,
                    "deterministic": deterministic}
        if (any(prior_config[key] != value for key, value in expected.items()) or
                prior_report["device"] != device or prior_rng["runtime"] != runtime or
                list(resume["rules"]) != names or resume["rules"] != definitions or
                _weights(names, prior_report["weights"]) != probs):
            raise ValueError("Incompatible resume configuration, rules, weights, device or runtime")
        previous_steps = prior_config["steps"]
        previous_elapsed = prior_report["elapsed_seconds"]
        previous_allocated = prior_report["peak_cuda_allocated_bytes"]
        previous_reserved = prior_report["peak_cuda_reserved_bytes"]
        if (isinstance(previous_elapsed, bool) or not isinstance(previous_elapsed, (int, float)) or
                not math.isfinite(previous_elapsed) or previous_elapsed < 0 or
                (device == "cpu" and (previous_allocated is not None or previous_reserved is not None)) or
                (device == "cuda" and any(type(value) is not int or value < 0
                                          for value in (previous_allocated, previous_reserved)))):
            raise ValueError("Checkpoint has invalid timing or peak-memory measurements")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = PilotUNet(width, timesteps).to(target)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    _, alpha_bar = schedule(timesteps, target)
    if resume is None:
        counts = {name: 0 for name in names}
        cooccurrence = {name: {event: 0 for event in names} for name in names}
        attempts_total = 0
    else:
        counts = copy.deepcopy(prior_report["condition_counts"])
        cooccurrence = copy.deepcopy(prior_report["realized_rule_counts_by_condition"])
        attempts_total = prior_report["sampling_attempts"]
        if (not isinstance(counts, dict) or set(counts) != set(names) or
                any(type(value) is not int or value < 0 for value in counts.values()) or
                sum(counts.values()) != previous_steps * batch_size or
                not isinstance(cooccurrence, dict) or set(cooccurrence) != set(names) or
                any(not isinstance(row, dict) or set(row) != set(names) or
                    any(type(value) is not int or value < 0 or value > counts[name] for value in row.values())
                    for name, row in cooccurrence.items()) or
                type(attempts_total) is not int or attempts_total < sum(counts.values())):
            raise ValueError("Checkpoint has invalid accumulated exposure counts")
        try:
            model.load_state_dict(resume["model"])
            ema.load_state_dict(resume["ema"])
            # Optimizer loading may retain CPU tensor aliases; the caller's
            # checkpoint must stay unchanged when resumed updates run.
            optimizer.load_state_dict(copy.deepcopy(resume["optimizer"]))
            rng.setstate(prior_rng["python"])
            torch.set_rng_state(prior_rng["torch_cpu"])
            if device == "cuda":
                states = prior_rng["torch_cuda"]
                if not isinstance(states, list) or len(states) != torch.cuda.device_count():
                    raise ValueError("CUDA RNG state does not match visible devices")
                torch.cuda.set_rng_state_all(states)
            elif prior_rng["torch_cuda"] is not None:
                raise ValueError("CPU checkpoint has unexpected CUDA RNG state")
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid checkpoint model, optimizer or random state: {exc}") from exc
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        torch.cuda.reset_peak_memory_stats(target)
    start = time.perf_counter()
    last_loss = None
    for _ in range(steps):
        indices = rng.choices(range(len(names)), weights=probs, k=batch_size)
        batch = []
        for index in indices:
            name = names[index]
            try:
                scene, attempts = render_conditioned(rng.getrandbits(63), rules[name], max_attempts)
            except ValueError as exc:
                raise ValueError(f"Condition {name!r} could not be sampled: {exc}") from exc
            counts[name] += 1
            attempts_total += attempts
            for event, predicates in rules.items():
                cooccurrence[name][event] += int(satisfies(scene.objects, predicates))
            pixels = np.array(scene.image, dtype=np.float32, copy=True)
            batch.append(torch.from_numpy(pixels).permute(2, 0, 1) / 127.5 - 1.0)
        clean = torch.stack(batch).to(target)
        t = torch.randint(timesteps, (batch_size,), device=target)
        noise = torch.randn_like(clean)
        abar = alpha_bar[t, None, None, None]
        noisy = abar.sqrt() * clean + (1 - abar).sqrt() * noise
        drop = torch.rand(batch_size, device=target) < dropout
        prediction = model(noisy, t, tokens[indices], drop)
        loss = F.mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite pilot loss; checkpoint not written")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for average, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                average.lerp_(parameter, 1.0 - ema_decay)
        last_loss = float(loss.detach().cpu())
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elapsed = previous_elapsed + time.perf_counter() - start
    config = {"steps": previous_steps + steps, "batch_size": batch_size, "seed": seed, "width": width,
              "timesteps": timesteps, "lr": lr, "dropout": dropout, "ema_decay": ema_decay,
              "max_attempts": max_attempts, "deterministic": deterministic}
    terminal_alpha_bar = float(alpha_bar[-1].cpu())
    result = {"config": config, "device": device, "last_loss": last_loss, "elapsed_seconds": elapsed,
              "terminal_alpha_bar": terminal_alpha_bar,
              "prior_approximation_warning": terminal_alpha_bar > 0.01,
              "peak_cuda_allocated_bytes": max(previous_allocated or 0, torch.cuda.max_memory_allocated(target)) if device == "cuda" else None,
              "peak_cuda_reserved_bytes": max(previous_reserved or 0, torch.cuda.max_memory_reserved(target)) if device == "cuda" else None,
              "condition_counts": counts, "realized_rule_counts_by_condition": cooccurrence,
              "sampling_attempts": attempts_total, "weights": dict(zip(names, probs, strict=True)),
              "note": ("Exploratory pilot only; no Stage 0 gate or model performance claim. "
                       "Timing includes online rendering and has no warmup; seeds do not guarantee "
                       "bitwise equality across hardware or PyTorch versions.")}
    checkpoint = {"schema_version": 3, "config": config, "rules": definitions,
                  "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                  "report": result, "rng": {"runtime": runtime, "python": rng.getstate(),
                                            "torch_cpu": torch.get_rng_state(),
                                            "torch_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None}}
    return checkpoint, result


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    if path.exists():
        raise ValueError("Checkpoint already exists; select a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pilot-", delete=False) as file:
            temporary = Path(file.name)
            torch.save(checkpoint, file)
        # Exclusive creation prevents overwriting an existing output.
        created = False
        try:
            with path.open("xb") as output:
                created = True
                with temporary.open("rb") as source:
                    import shutil
                    shutil.copyfileobj(source, output)
        except Exception:
            if created:
                path.unlink(missing_ok=True)
            raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@torch.no_grad()
def sample(checkpoint: dict, rule: str, *, seed: int, device: str, guidance: float) -> Image.Image:
    require_standard_predictor(checkpoint)
    if seed < 0 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("seed and guidance must be finite and nonnegative")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Requested device is unavailable")
    config = checkpoint["config"]
    _configure_determinism(config.get("deterministic", False), device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device is unavailable")
    target = torch.device(device)
    definitions = checkpoint["rules"]
    rules = load_rules(definitions)
    if rule not in rules:
        raise ValueError("Unknown rule")
    names = list(rules)
    tokens = encode_rules(rules)[names.index(rule):names.index(rule) + 1].to(target)
    model = PilotUNet(config["width"], config["timesteps"]).to(target)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    beta, alpha_bar = schedule(config["timesteps"], target)
    if float(alpha_bar[-1].cpu()) > 0.01:
        warnings.warn("Terminal signal is not close to zero; sampling from Gaussian noise is only a smoke approximation",
                      RuntimeWarning, stacklevel=2)
    generator = torch.Generator(device=target).manual_seed(seed)
    x = torch.randn((1, 3, 64, 64), device=target, generator=generator)
    for step in range(config["timesteps"] - 1, -1, -1):
        t = torch.tensor([step], device=target)
        conditional = model(x, t, tokens)
        if guidance != 1.0:
            unconditional = model(x, t, tokens, torch.ones(1, dtype=torch.bool, device=target))
            conditional = unconditional + guidance * (conditional - unconditional)
        x = (x - beta[step] / (1 - alpha_bar[step]).sqrt() * conditional) / (1 - beta[step]).sqrt()
        if step:
            variance = beta[step] * (1 - alpha_bar[step - 1]) / (1 - alpha_bar[step])
            x = x + variance.sqrt() * torch.randn(x.shape, device=target, generator=generator)
    image = ((x[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory online conditional diffusion pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
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
    fit.add_argument("--deterministic", action="store_true", help="require deterministic PyTorch algorithms")
    resume_parser = sub.add_parser("resume", help="run additional updates from a version 3 checkpoint")
    resume_parser.add_argument("--checkpoint", type=Path, required=True)
    resume_parser.add_argument("--steps", type=int, required=True, help="number of additional updates")
    resume_parser.add_argument("--output", type=Path, required=True, help="new checkpoint path")
    resume_parser.add_argument("--device", choices=("cpu", "cuda"), help="must match checkpoint device")
    resume_parser.add_argument("--deterministic", action="store_true", default=None,
                               help="require a deterministic checkpoint; mode is inherited by default")
    generate = sub.add_parser("sample")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--rule", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    generate.add_argument("--guidance", type=float, default=1.0)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output already exists; select a new path")
        if args.command == "train":
            definitions = _load_json(args.rules)
            weights = _load_json(args.weights) if args.weights else None
            checkpoint, result = train(definitions, weights, steps=args.steps, batch_size=args.batch_size,
                                       seed=args.seed, device=args.device, width=args.width,
                                       timesteps=args.timesteps, lr=args.lr, dropout=args.dropout,
                                       ema_decay=args.ema_decay, max_attempts=args.max_attempts,
                                       deterministic=args.deterministic)
            save_checkpoint(args.output, checkpoint)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "resume":
            old = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            config, prior_report, _ = _resume_parts(old)
            device = args.device or prior_report["device"]
            checkpoint, result = train(old["rules"], prior_report["weights"], steps=args.steps,
                                       batch_size=config["batch_size"], seed=config["seed"], device=device,
                                       width=config["width"], timesteps=config["timesteps"], lr=config["lr"],
                                       dropout=config["dropout"], ema_decay=config["ema_decay"],
                                       max_attempts=config["max_attempts"],
                                       deterministic=(config["deterministic"] if args.deterministic is None
                                                      else args.deterministic), resume=old)
            save_checkpoint(args.output, checkpoint)
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            image = sample(checkpoint, args.rule, seed=args.seed, device=args.device, guidance=args.guidance)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as file:
                image.save(file, format="PNG")
            print(f"Wrote {args.output}")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
