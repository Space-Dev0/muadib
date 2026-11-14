"""Warm-start a MOSAIC PilotUNet velocity probe from an earlier checkpoint.

This module is deliberately separate from :mod:`exposure.mosaic_pilotunet_probe`.
It restores the model, EMA and AdamW optimizer state from a schema-1
``mosaic_pilotunet_velocity`` checkpoint.  That source checkpoint did **not**
store random-number-generator state, so a warm start from it is **not** an exact
resume: fresh Python, NumPy and PyTorch streams are derived from a new seed, and
that limitation is recorded in every report and checkpoint.

Every checkpoint written by this module instead contains the full optimizer
state plus Python, NumPy, PyTorch CPU/CUDA and explicit-generator RNG state, so
subsequent runs from *these* checkpoints can be exact on the same device type and
compatible runtime.  Bitwise equality across hardware, drivers or PyTorch
versions is not guaranteed.

The module is exploratory.  It is not a Stage 0 gate decision, a validated
quality metric, or a claim of MOSAIC replication.  Existing datasets,
checkpoints and other probe files are never modified; output directories and
checkpoints must be new paths.
"""

import argparse
import copy
import hashlib
import io
import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from exposure import mosaic_pilotunet_probe as probe
from exposure import pilot_train
from exposure.velocity_probe import velocity_target

SOURCE_KIND = "mosaic_pilotunet_velocity"
CHECKPOINT_KIND = "mosaic_pilotunet_velocity_warmstart"
SCHEMA_VERSION = 1
RNG_STREAMS = ("time", "noise", "drop")
RNG_KEYS = {"python_random", "torch_cpu", "torch_cuda", "generators", "numpy",
            "seed", "stream_seeds", "source", "not_exact_resume", "captured_at_step",
            "device", "note"}
NOT_EXACT_REASON = ("The source checkpoint did not persist Python, NumPy or PyTorch "
                    "RNG state; fresh streams were derived from the new seed. This is a "
                    "warm start, not an exact resume.")
RNG_CAVEAT = ("Exact continuation is only expected on the same device type and a "
              "compatible Python/PyTorch runtime; bitwise equality across hardware, "
              "drivers or PyTorch versions is not guaranteed.")


@dataclass
class RngStreams:
    """Independent random streams used by the training loop."""

    python_random: random.Random
    time: torch.Generator
    noise: torch.Generator
    drop: torch.Generator
    seed: int
    stream_seeds: dict
    source: str
    not_exact_resume: bool


def _derive_seed(seed: int, name: str) -> int:
    """Derive an independent 63-bit stream seed from a base seed and a name."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    digest = hashlib.sha256(f"{seed}:{name}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _numpy_state() -> dict:
    """Return the NumPy global RNG state using only checkpoint-safe primitives."""
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"name": str(name), "keys": [int(value) for value in keys],
            "position": int(position), "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached)}


def _set_numpy_state(state: dict) -> None:
    np.random.set_state((state["name"], np.array(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"],
                         state["cached_gaussian"]))


def fresh_rng(seed: int, device: str) -> RngStreams:
    """Create freshly seeded streams; explicitly not an exact resume."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    stream_seeds = {name: _derive_seed(seed, name)
                    for name in ("index", *RNG_STREAMS)}
    generators = {name: torch.Generator(device=device).manual_seed(stream_seeds[name])
                  for name in RNG_STREAMS}
    return RngStreams(python_random=random.Random(stream_seeds["index"]),
                      time=generators["time"], noise=generators["noise"],
                      drop=generators["drop"], seed=seed, stream_seeds=stream_seeds,
                      source="fresh_seed", not_exact_resume=True)


def capture_rng(rng: RngStreams, device: str, *, captured_at_step: int) -> dict:
    """Snapshot every RNG stream used by training in one checkpointable dict."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    cuda_states = None
    if device == "cuda":
        cuda_states = [state.detach().cpu().clone()
                       for state in torch.cuda.get_rng_state_all()]
    return {
        "python_random": rng.python_random.getstate(),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": cuda_states,
        "generators": {name: getattr(rng, name).get_state().detach().cpu().clone()
                       for name in RNG_STREAMS},
        "numpy": _numpy_state(),
        "seed": rng.seed,
        "stream_seeds": dict(rng.stream_seeds),
        "source": rng.source,
        "not_exact_resume": rng.not_exact_resume,
        "captured_at_step": int(captured_at_step),
        "device": device,
        "note": ("Complete training RNG snapshot for exact continuation on the same "
                 "device type and compatible runtime. " + RNG_CAVEAT),
    }


def _as_tuple(value):
    """Recursively convert lists to tuples for random.Random.setstate."""
    if isinstance(value, (list, tuple)):
        return tuple(_as_tuple(item) for item in value)
    return value


def _validate_rng_state(state: dict) -> None:
    """Check the snapshot shape without mutating any process-wide RNG state."""
    if not isinstance(state, dict) or not RNG_KEYS <= state.keys():
        raise ValueError("Checkpoint lacks complete RNG state")
    if not isinstance(state["generators"], dict) or set(state["generators"]) != set(RNG_STREAMS):
        raise ValueError("Checkpoint lacks explicit generator RNG state")
    if not torch.is_tensor(state["torch_cpu"]):
        raise ValueError("Checkpoint CPU RNG state is not a tensor")
    if any(not torch.is_tensor(value) for value in state["generators"].values()):
        raise ValueError("Checkpoint generator RNG state is not a tensor")
    if state.get("device") not in ("cpu", "cuda"):
        raise ValueError("Checkpoint RNG state has an invalid device")
    if not isinstance(state["numpy"], dict) or "keys" not in state["numpy"]:
        raise ValueError("Checkpoint NumPy RNG state is malformed")


def restore_rng(state: dict, device: str) -> RngStreams:
    """Restore a :func:`capture_rng` snapshot and rebuild the stream objects."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    _validate_rng_state(state)
    python_random = random.Random()
    python_random.setstate(_as_tuple(state["python_random"]))
    torch.set_rng_state(state["torch_cpu"].detach().cpu().clone())
    if device == "cuda":
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state does not match visible devices")
        torch.cuda.set_rng_state_all(cuda_states)
    elif state["torch_cuda"] is not None:
        raise ValueError("Cannot exactly resume CUDA RNG state on CPU; use the same device type")
    _set_numpy_state(state["numpy"])
    generators = {}
    for name in RNG_STREAMS:
        generator = torch.Generator(device=device)
        generator.set_state(state["generators"][name].detach().cpu().clone())
        generators[name] = generator
    return RngStreams(python_random=python_random, time=generators["time"],
                      noise=generators["noise"], drop=generators["drop"],
                      seed=int(state["seed"]), stream_seeds=dict(state["stream_seeds"]),
                      source="restored", not_exact_resume=False)


def _rng_fingerprint(state: dict) -> str:
    """Fingerprint only the RNG streams, excluding descriptive metadata."""
    core = {key: state[key] for key in
            ("python_random", "torch_cpu", "torch_cuda", "generators", "numpy")}
    buffer = io.BytesIO()
    torch.save(core, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _state_fingerprint(state) -> str:
    """Stable fingerprint over tensors and plain optimizer/state values."""
    digest = hashlib.sha256()

    def update(value) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode("utf-8"))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                update(item)
        else:
            digest.update(repr(value).encode("utf-8"))

    update(state)
    return digest.hexdigest()


def load_source_checkpoint(path: Path) -> dict:
    """Load and validate the legacy velocity checkpoint that is warm-started."""
    checkpoint = probe.load_checkpoint(Path(path))
    if checkpoint.get("kind") != SOURCE_KIND:
        raise ValueError("Expected a mosaic_pilotunet_velocity source checkpoint")
    if "rng" in checkpoint:
        raise ValueError("Source unexpectedly has RNG state; use the resume mode instead")
    required = {"model", "ema", "optimizer", "config", "rules", "label_to_row",
                "report", "provenance"}
    if not required <= checkpoint.keys():
        raise ValueError("Source checkpoint lacks model, EMA or optimizer state")
    return checkpoint


def load_warmstart_checkpoint(path: Path) -> dict:
    """Load and validate a checkpoint written by this module."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    if (checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("kind") != CHECKPOINT_KIND):
        raise ValueError("Expected a schema-1 mosaic PilotUNet warm-start checkpoint")
    diagnostic = checkpoint.get("diagnostic")
    if (not isinstance(diagnostic, dict) or diagnostic.get("prediction") != "velocity"
            or diagnostic.get("resumable") is not True):
        raise ValueError("Checkpoint does not declare a resumable velocity model")
    required = {"config", "rules", "label_to_row", "model", "ema", "optimizer",
                "rng", "report", "warm_start", "source_checkpoint", "provenance"}
    if not required <= checkpoint.keys():
        raise ValueError("Checkpoint lacks required warm-start resume state")
    _validate_rng_state(checkpoint["rng"])
    return checkpoint


def _optimizer_lr(checkpoint: dict) -> float:
    value = checkpoint["config"].get("lr")
    if value is not None:
        return float(value)
    groups = checkpoint["optimizer"].get("param_groups")
    if not isinstance(groups, list) or not groups or "lr" not in groups[0]:
        raise ValueError("Cannot determine the source learning rate")
    return float(groups[0]["lr"])


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_training_state(checkpoint: dict, device: torch.device,
                           *, lr: float) -> tuple[object, object, torch.optim.Optimizer]:
    """Rebuild PilotUNet, EMA and AdamW and load source tensors strictly."""
    configuration = checkpoint["config"]
    width, timesteps = configuration.get("width"), configuration.get("timesteps")
    if type(width) is not int or width < 4 or width % 4:
        raise ValueError("Source checkpoint has an invalid width")
    if type(timesteps) is not int or timesteps < 2:
        raise ValueError("Source checkpoint has an invalid timestep count")
    model = pilot_train.PilotUNet(width, timesteps)
    ema = copy.deepcopy(model).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Copy so the source checkpoint stays unchanged when resumed updates run.
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.to(device)
    ema.to(device)
    _move_optimizer_state(optimizer, device)
    return model, ema, optimizer


def _source_info(path: Path, digest: str, checkpoint: dict, *, exact: bool) -> dict:
    report = checkpoint["report"]
    return {
        "path": str(Path(path).resolve()) if Path(path).exists() else str(path),
        "sha256": digest,
        "kind": checkpoint.get("kind"),
        "schema_version": checkpoint.get("schema_version"),
        "reported_step": report.get("step", report.get("cumulative_step")),
        "config": copy.deepcopy(checkpoint["config"]),
        "restored": ["model", "ema", "optimizer"] + (["rng"] if exact else []),
        "exact_resume": bool(exact),
        "not_exact_resume": not exact,
        "reason": None if exact else NOT_EXACT_REASON,
    }


def _make_checkpoint(*, config: dict, definitions: dict, label_to_row: dict,
                     model, ema, optimizer, rng_state: dict, step: int,
                     start_step: int, loss_mean: float, elapsed: float,
                     source_info: dict, provenance: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "cell": list(probe.CELL),
        "config": copy.deepcopy(config),
        "rules": copy.deepcopy(definitions),
        "label_to_row": dict(label_to_row),
        "model": {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()},
        "ema": {key: value.detach().cpu().clone()
                for key, value in ema.state_dict().items()},
        "optimizer": probe._cpu_state(optimizer.state_dict()),
        "rng": copy.deepcopy(rng_state),
        "report": {
            "start_step": int(start_step),
            "steps_added": int(step - start_step),
            "cumulative_step": int(step),
            "loss_mean_last_100": float(loss_mean),
            "elapsed_seconds": float(elapsed),
            "resume_mode": config["resume_mode"],
            "not_exact_resume": bool(config["not_exact_resume"]),
            "rng_fingerprint": _rng_fingerprint(rng_state),
            "note": ("Exploratory warm-start checkpoint; not a Stage 0 gate or "
                     "model-performance claim."),
        },
        "diagnostic": {
            "kind": CHECKPOINT_KIND,
            "prediction": "velocity",
            "sampler": "ddpm_velocity_to_noise",
            "requires_custom_sampler": True,
            "resumable": True,
            "optimizer_state": "included",
            "rng_state": "included",
            "note": ("Velocity weights require velocity-to-noise conversion during "
                     "sampling. " + RNG_CAVEAT),
        },
        "warm_start": copy.deepcopy(source_info),
        "source_checkpoint": copy.deepcopy(source_info),
        "provenance": copy.deepcopy(provenance),
    }


def _provenance(device: str, dataset: Path, data_info: dict, config: dict,
                definitions: dict, source_info: dict) -> dict:
    base = probe._provenance(device, dataset, data_info["manifest_sha256"],
                             data_info["records_sha256"], config, definitions,
                             data_info["records"])
    base["runtime"]["code_sha256"]["mosaic_pilotunet_warmstart"] = probe._sha256_path(
        Path(__file__))
    base["source_checkpoint"] = copy.deepcopy(source_info)
    base["note"] = ("Provenance for an exploratory warm start. The source checkpoint "
                    "lacked RNG state, so this run is not an exact resume. New "
                    "checkpoints contain optimizer and RNG state. " + RNG_CAVEAT)
    return base


def _snapshot(*, output: Path, step: int, start_step: int, config: dict,
              definitions: dict, label_to_row: dict, model, ema, optimizer,
              rng: RngStreams, tokens: torch.Tensor, names: list[str],
              beta: torch.Tensor, alpha_bar: torch.Tensor, preview_seeds: list[int],
              device: str, losses: list[float], wall_seconds: float,
              source_info: dict, provenance: dict) -> dict:
    mean = float(np.mean(losses[-100:])) if losses else float("nan")
    rng_state = capture_rng(rng, device, captured_at_step=step)
    checkpoint_path = output / f"warmstart-{step:06d}.pt"
    checkpoint = _make_checkpoint(config=config, definitions=definitions,
                                  label_to_row=label_to_row, model=model, ema=ema,
                                  optimizer=optimizer, rng_state=rng_state, step=step,
                                  start_step=start_step, loss_mean=mean,
                                  elapsed=wall_seconds, source_info=source_info,
                                  provenance=provenance)
    pilot_train.save_checkpoint(checkpoint_path, checkpoint)
    previews = None
    if preview_seeds:
        previews = probe._preview(output, ema, tokens, names, beta, alpha_bar,
                                  step=step, seeds=preview_seeds, device=device)
    return {
        "step": step,
        "steps_added": step - start_step,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": probe._sha256_path(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "last_loss": float(losses[-1]) if losses else float("nan"),
        "loss_mean_last_100": mean,
        "wall_seconds_since_start": wall_seconds,
        "model_sha256": _state_fingerprint(model.state_dict()),
        "ema_sha256": _state_fingerprint(ema.state_dict()),
        "optimizer_sha256": _state_fingerprint(optimizer.state_dict()),
        "rng_fingerprint": _rng_fingerprint(rng_state),
        "rng_captured_at_step": step,
        "optimizer_state_included": True,
        "rng_state_included": True,
        "previews": previews,
    }


def _validate_data_against_source(source: dict, data_info: dict) -> None:
    source_data = source.get("provenance", {}).get("dataset")
    if not isinstance(source_data, dict):
        raise ValueError("Source checkpoint has no dataset provenance")
    if source_data.get("manifest_sha256") != data_info["manifest_sha256"]:
        raise ValueError("Dataset manifest does not match the source checkpoint")
    if source_data.get("records_sha256") != data_info["records_sha256"]:
        raise ValueError("Dataset records do not match the source checkpoint")


def _resolve_config(source: dict, *, mode: str, seed, steps: int, batch_size, lr,
                    dropout, ema_decay, deterministic, preview_seeds: list[int],
                    snapshot_steps) -> dict:
    source_config = source["config"]
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if mode == "resume":
        if seed is not None and seed != source_config["seed"]:
            raise ValueError("Resume must reuse the checkpoint seed for RNG restoration")
        seed = source_config["seed"]
    elif type(seed) is not int or seed < 0:
        raise ValueError("Warm start requires a nonnegative --seed")
    batch_size = source_config["batch_size"] if batch_size is None else batch_size
    lr = _optimizer_lr(source) if lr is None else lr
    dropout = source_config["dropout"] if dropout is None else dropout
    ema_decay = source_config["ema_decay"] if ema_decay is None else ema_decay
    deterministic = (bool(source_config.get("deterministic", False))
                     if deterministic is None else bool(deterministic))
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if mode == "resume":
        expected = {"batch_size": batch_size, "lr": float(lr),
                    "dropout": float(dropout), "ema_decay": float(ema_decay),
                    "deterministic": deterministic}
        actual = {"batch_size": source_config["batch_size"],
                  "lr": float(source_config["lr"]),
                  "dropout": source_config["dropout"],
                  "ema_decay": source_config["ema_decay"],
                  "deterministic": bool(source_config.get("deterministic", False))}
        if expected != actual:
            raise ValueError("Exact resume requires the checkpoint's batch/lr/dropout/EMA/"
                             "deterministic settings; omit overrides or match them")
        if preview_seeds:
            raise ValueError("Exact resume does not support preview side effects; omit previews")
    if not isinstance(preview_seeds, list) or any(
            type(value) is not int or value < 0 for value in preview_seeds):
        raise ValueError("preview_seeds must be a list of nonnegative integers")
    if len(set(preview_seeds)) != len(preview_seeds):
        raise ValueError("preview_seeds must be unique")
    start_step = int(source["report"].get("cumulative_step",
                                          source["report"].get("step", 0)))
    if start_step < 0:
        raise ValueError("Source checkpoint has an invalid step count")
    if snapshot_steps is None:
        snapshot_steps = sorted({start_step + max(1, steps // 2), start_step + steps})
    if (not isinstance(snapshot_steps, (list, tuple, set)) or not snapshot_steps
            or any(type(value) is not int or not start_step < value <= start_step + steps
                   for value in snapshot_steps)):
        raise ValueError("snapshot_steps must be integers within the new step range")
    snapshot_steps = sorted(set(snapshot_steps))
    return {
        "cell": list(probe.CELL),
        "resume_mode": mode,
        "not_exact_resume": mode == "warmstart",
        "seed": seed,
        "new_steps": steps,
        "batch_size": batch_size,
        "width": source_config["width"],
        "timesteps": source_config["timesteps"],
        "lr": float(lr),
        "dropout": float(dropout),
        "ema_decay": float(ema_decay),
        "deterministic": deterministic,
        "preview_seeds": list(preview_seeds),
        "snapshot_steps": snapshot_steps,
        "start_step": start_step,
        "sampler": {"name": "ddpm_velocity_to_noise",
                    "steps": source_config["timesteps"], "guidance": 1.0},
    }


def _train(source: dict, *, source_info: dict, images: torch.Tensor, colors: list[str],
           data_info: dict, output: Path, config: dict, provenance: dict,
           definitions: dict, label_to_row: dict, device: str) -> dict:
    """Run bounded updates and write periodic checkpoints with full resume state."""
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; select a new directory")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, 64, 64)")
    if len(colors) != images.shape[0] or any(color not in probe.COLORS for color in colors):
        raise ValueError("colors must label every image with red or blue")
    resolved = probe._resolve_device(device)
    pilot_train._configure_determinism(config["deterministic"], resolved)
    target = torch.device(resolved)
    tokens, names, loaded_definitions, loaded_rows = probe._token_table(target)
    if definitions != loaded_definitions or label_to_row != loaded_rows:
        raise ValueError("Cell rules or label mapping changed between runs")
    if list(source.get("cell", probe.CELL)) != list(probe.CELL):
        raise ValueError("Source checkpoint was trained on a different cell")
    if source["rules"] != definitions:
        raise ValueError("Source checkpoint rules differ from the low/balanced cell")
    if source.get("label_to_row") != label_to_row:
        raise ValueError("Source checkpoint label mapping differs from the low/balanced cell")

    output.mkdir(parents=True, exist_ok=False)
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    started = time.perf_counter()
    try:
        model, ema, optimizer = restore_training_state(
            source, target, lr=config["lr"])
        restored = {
            "source_model_sha256": _state_fingerprint(source["model"]),
            "restored_model_sha256": _state_fingerprint(model.state_dict()),
            "source_ema_sha256": _state_fingerprint(source["ema"]),
            "restored_ema_sha256": _state_fingerprint(ema.state_dict()),
            "source_optimizer_sha256": _state_fingerprint(source["optimizer"]),
            "restored_optimizer_sha256": _state_fingerprint(optimizer.state_dict()),
        }
        restored["matches_source"] = (
            restored["source_model_sha256"] == restored["restored_model_sha256"]
            and restored["source_ema_sha256"] == restored["restored_ema_sha256"]
            and restored["source_optimizer_sha256"] == restored["restored_optimizer_sha256"])
        if not restored["matches_source"]:
            raise RuntimeError("Restored model/EMA/optimizer does not match the source checkpoint")

        if config["resume_mode"] == "warmstart":
            rng = fresh_rng(config["seed"], resolved)
            rng_restore = {"mode": "fresh_seed", "not_exact_resume": True,
                           "reason": NOT_EXACT_REASON}
        else:
            rng = restore_rng(source["rng"], resolved)
            resumed_fingerprint = _rng_fingerprint(
                capture_rng(rng, resolved, captured_at_step=config["start_step"]))
            expected = source["report"].get("rng_fingerprint")
            if expected is not None and resumed_fingerprint != expected:
                raise RuntimeError("Restored RNG state does not match the checkpoint")
            rng_restore = {"mode": "restored", "not_exact_resume": False,
                           "rng_fingerprint": resumed_fingerprint,
                           "matches_checkpoint": expected is None or resumed_fingerprint == expected}

        beta, alpha_bar = pilot_train.schedule(config["timesteps"], target)
        images = images.to(target)
        token_indices = torch.tensor([label_to_row[color] for color in colors],
                                     dtype=torch.long, device=target)
        losses: list[float] = []
        loss_trace: list[dict] = []
        snapshots: list[dict] = []
        if resolved == "cuda":
            torch.cuda.synchronize(target)
            torch.cuda.reset_peak_memory_stats(target)
        optimizer_seconds = 0.0
        for offset in range(1, config["new_steps"] + 1):
            step = config["start_step"] + offset
            step_started = time.perf_counter()
            index_list = [rng.python_random.randrange(len(colors))
                          for _ in range(config["batch_size"])]
            indices = torch.tensor(index_list, dtype=torch.long, device=target)
            clean = images[indices]
            t = torch.randint(config["timesteps"], (config["batch_size"],),
                              device=target, generator=rng.time)
            noise = torch.randn(clean.shape, device=target, generator=rng.noise)
            alpha = alpha_bar[t, None, None, None]
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            drop = torch.rand(config["batch_size"], device=target,
                              generator=rng.drop) < config["dropout"]
            prediction = model(noisy, t, tokens[token_indices[indices]], drop)
            loss = F.mse_loss(prediction, velocity_target(clean, noise, alpha))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite velocity loss; no report written")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters(),
                                              strict=True):
                    average.lerp_(parameter, 1.0 - config["ema_decay"])
            optimizer_seconds += time.perf_counter() - step_started
            value = float(loss.detach().cpu())
            losses.append(value)
            if offset == 1 or offset % 100 == 0 or step in config["snapshot_steps"]:
                loss_trace.append({"step": step, "loss": value})
            if step in config["snapshot_steps"]:
                if resolved == "cuda":
                    torch.cuda.synchronize(target)
                wall_seconds = time.perf_counter() - started
                snapshots.append(_snapshot(
                    output=output, step=step, start_step=config["start_step"],
                    config=config, definitions=definitions,
                    label_to_row=label_to_row, model=model, ema=ema,
                    optimizer=optimizer, rng=rng, tokens=tokens, names=names,
                    beta=beta, alpha_bar=alpha_bar,
                    preview_seeds=config["preview_seeds"], device=resolved,
                    losses=losses, wall_seconds=wall_seconds,
                    source_info=source_info, provenance=provenance))
        if resolved == "cuda":
            torch.cuda.synchronize(target)
        total_elapsed = time.perf_counter() - started
        peak_allocated = (torch.cuda.max_memory_allocated(target)
                          if resolved == "cuda" else None)
        peak_reserved = (torch.cuda.max_memory_reserved(target)
                         if resolved == "cuda" else None)
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND + "_report",
            "mode": config["resume_mode"],
            "output": str(output),
            "cell": list(probe.CELL),
            "config": config,
            "source_checkpoint": source_info,
            "warm_start": {
                "exact_resume": not config["not_exact_resume"],
                "not_exact_resume": config["not_exact_resume"],
                "reason": None if not config["not_exact_resume"] else NOT_EXACT_REASON,
                "restored_state": copy.deepcopy(restored),
                "rng": copy.deepcopy(rng_restore),
                "caveat": RNG_CAVEAT,
            },
            "device": {"requested": device, "resolved": resolved},
            "data": copy.deepcopy(data_info),
            "rules": copy.deepcopy(definitions),
            "label_to_row": dict(label_to_row),
            "provenance": copy.deepcopy(provenance),
            "training": {
                "start_step": config["start_step"],
                "steps_added": config["new_steps"],
                "cumulative_step": config["start_step"] + config["new_steps"],
                "optimizer_seconds": optimizer_seconds,
                "total_elapsed_seconds_including_previews": total_elapsed,
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "loss_mean_last_100": float(np.mean(losses[-100:])),
                "loss_trace": loss_trace,
            },
            "snapshots": snapshots,
            "limitations": [
                ("Warm start only: the source checkpoint did not store RNG state, so this is not an exact resume."
                 if config["not_exact_resume"] else
                 "Exact resume from a warm-start checkpoint on the same device type and compatible runtime."),
                "Single cell (low/balanced), a short bounded budget and one architecture scale; no ablation.",
                "The dataset is a 200-image synthetic feasibility set, not the MOSAIC benchmark.",
                "No held-out likelihood, shape metric or object-level validation; previews are qualitative.",
                "New checkpoints store optimizer and RNG state, but exactness is only expected on the same device type and compatible runtime.",
                "This exploratory run is not a Stage 0 gate decision or a claim of MOSAIC replication.",
            ],
            "note": (("Short warm-start from an earlier velocity probe checkpoint. It verifies "
                      "state restoration and checkpoint provenance; it does not establish "
                      "sample quality or Stage 0 passage.")
                     if config["not_exact_resume"] else
                     ("Short exact resume from a warm-start checkpoint. It verifies optimizer "
                      "and RNG restoration; it does not establish sample quality or Stage 0 "
                      "passage.")),
        }
        with (output / "report.json").open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        return report
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(),
                   "note": "Partial outputs are preserved; no report.json was written."}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n",
                                              encoding="utf-8")
        raise


def run(source_path: Path, dataset: Path, output: Path, *, mode: str, seed=None,
        steps: int = 2, batch_size=None, lr=None, dropout=None, ema_decay=None,
        device: str = "auto", preview_seeds=None, snapshot_steps=None,
        deterministic=None) -> dict:
    """Validate inputs, load the dataset, then run a bounded warm start/resume."""
    if mode not in ("warmstart", "resume"):
        raise ValueError("mode must be warmstart or resume")
    source_path = Path(source_path).resolve(strict=True)
    source_digest = probe._sha256_path(source_path)
    if mode == "warmstart":
        source = load_source_checkpoint(source_path)
    else:
        source = load_warmstart_checkpoint(source_path)
    data_info = _load_dataset(dataset)
    _validate_data_against_source(source, data_info)
    config = _resolve_config(source, mode=mode, seed=seed, steps=steps,
                             batch_size=batch_size, lr=lr, dropout=dropout,
                             ema_decay=ema_decay, deterministic=deterministic,
                             preview_seeds=list(preview_seeds or []),
                             snapshot_steps=snapshot_steps)
    source_info = _source_info(source_path, source_digest, source,
                               exact=mode == "resume")
    definitions = source["rules"]
    device_resolved = probe._resolve_device(device)
    provenance = _provenance(device_resolved, dataset, data_info, config,
                             definitions, source_info)
    return _train(source, source_info=source_info, images=data_info.pop("images"),
                  colors=data_info.pop("colors"), data_info=data_info,
                  output=Path(output), config=config, provenance=provenance,
                  definitions=definitions, label_to_row=source["label_to_row"],
                  device=device)


def _load_dataset(dataset: Path) -> dict:
    images, colors, manifest_hash, records, data_hash = probe.load_cell(Path(dataset))
    return {
        "path": str(Path(dataset).resolve()),
        "manifest_sha256": manifest_hash,
        "records_sha256": data_hash,
        "training_images": len(records),
        "label_counts": {color: colors.count(color) for color in probe.COLORS},
        "records": records,
        "images": images,
        "colors": colors,
    }


def sample(checkpoint: dict, color: str, *, seed: int, device: str = "auto"):
    """Sample one target colour from a warm-start checkpoint with EMA weights."""
    if (checkpoint.get("kind") != CHECKPOINT_KIND
            or checkpoint.get("schema_version") != SCHEMA_VERSION):
        raise ValueError("Expected a mosaic PilotUNet warm-start checkpoint")
    if color not in probe.COLORS:
        raise ValueError("color must be red or blue")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = checkpoint["config"]
    resolved = probe._resolve_device(device)
    pilot_train._configure_determinism(bool(config.get("deterministic", False)), resolved)
    target = torch.device(resolved)
    parsed = pilot_train.load_rules(checkpoint["rules"])
    names = list(parsed)
    if color not in names:
        raise ValueError("Checkpoint does not contain the requested colour rule")
    index = names.index(color)
    tokens = pilot_train.encode_rules(parsed)[index:index + 1].to(target)
    model = pilot_train.PilotUNet(config["width"], config["timesteps"]).to(target)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    beta, alpha_bar = pilot_train.schedule(config["timesteps"], target)
    return probe.sample_image(model, tokens, beta, alpha_bar, seed=seed,
                              device=resolved)


def _add_common(parser: argparse.ArgumentParser, *, resume: bool) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2 if not resume else 1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preview-seeds", type=int, nargs="+", default=[])
    parser.add_argument("--snapshot-steps", type=int, nargs="+", default=None)
    parser.add_argument("--deterministic", action="store_true", default=None)


def _summary(report: dict) -> dict:
    return {
        "mode": report["mode"],
        "output": report["output"],
        "source_checkpoint": report["source_checkpoint"],
        "warm_start": report["warm_start"],
        "device": report["device"],
        "config": report["config"],
        "training": report["training"],
        "snapshots": [
            {"step": snapshot["step"],
             "checkpoint": snapshot["checkpoint"],
             "checkpoint_sha256": snapshot["checkpoint_sha256"],
             "last_loss": snapshot["last_loss"],
             "loss_mean_last_100": snapshot["loss_mean_last_100"],
             "optimizer_state_included": snapshot["optimizer_state_included"],
             "rng_state_included": snapshot["rng_state_included"],
             "rng_fingerprint": snapshot["rng_fingerprint"],
             "model_sha256": snapshot["model_sha256"]}
            for snapshot in report["snapshots"]
        ],
        "limitations": report["limitations"],
        "note": report["note"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    warmstart = commands.add_parser("warmstart",
                                    help="restore model/EMA/optimizer and use fresh RNG")
    _add_common(warmstart, resume=False)
    resume = commands.add_parser("resume",
                                 help="exactly resume a checkpoint written by this module")
    _add_common(resume, resume=True)
    generate = commands.add_parser("sample", help="sample a warm-start checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--color", choices=probe.COLORS, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        if args.command == "sample":
            if args.output.exists():
                raise ValueError("Output already exists; select a new path")
            checkpoint = load_warmstart_checkpoint(args.checkpoint)
            image = sample(checkpoint, args.color, seed=args.seed, device=args.device)
            digest = probe._save_png(args.output, image)
            print(json.dumps({"output": str(args.output), "sha256": digest,
                              "color": args.color, "seed": args.seed,
                              "checkpoint_sha256": probe._sha256_path(args.checkpoint),
                              "note": "Exploratory sample; not a Stage 0 result."},
                             indent=2))
        else:
            report = run(args.source, args.dataset, args.output, mode=args.command,
                         seed=args.seed, steps=args.steps, batch_size=args.batch_size,
                         lr=args.lr, dropout=args.dropout, ema_decay=args.ema_decay,
                         device=args.device, preview_seeds=args.preview_seeds,
                         snapshot_steps=args.snapshot_steps,
                         deterministic=args.deterministic)
            print(json.dumps(_summary(report), indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
