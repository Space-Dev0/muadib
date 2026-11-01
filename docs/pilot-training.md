# Exploratory pilot training

**Preliminary and non-confirmatory. Stage 0 has not passed.** This is an
instrument-development pilot, not a research result. No actual CUDA model
training has been run, and no model output has been validated.

## What the pilot implements

`exposure.pilot_train` is a small online conditional diffusion instrument:

- The renderer produces fresh annotated 64×64 scenes on every training step;
  there is no fixed dataset.
- `PilotUNet` predicts noise with time embeddings and predicate-slot tokens
  describing predicate kind, subject slot, object slot, and attribute value.
  One cross-attention block operates at 16×16. A learned null token is always
  exposed for an empty or fully dropped condition.
- Conditions are dropped per sample for classifier-free training, then a null
  token is exposed so attention never sees an all-padding mask.
- Training uses AdamW and an EMA copy of the model. Sampling uses the EMA
  weights and a fixed DDPM sampler with a linear beta schedule.
- No mixed precision is implemented.

## Install the training extra

```sh
uv sync --extra train
```

On Windows and Linux, this installs `torch==2.9.1` from the CUDA 12.8 index.
The download is large. The repository itself still runs its normal CPU checks
without the training extra.

On the prepared Windows laptop, `uv` is inside the project virtual environment
rather than on PATH. Once the repository is present, use
`.venv\Scripts\python.exe -m uv sync --inexact --extra train` in PowerShell to
preserve that `uv` installation, then invoke the pilot with
`.venv\Scripts\python.exe -m exposure.pilot_train` and the arguments below.

## CPU mechanics smoke

Use a new output path; the CLI refuses to overwrite an existing checkpoint or
image.

```sh
uv run --extra train python -m exposure.pilot_train train --rules examples/exposure_rules.json --weights examples/uniform_weights.json --output outputs/pilot-smoke-20261101-01/pilot.pt --steps 1 --batch-size 1 --seed 0 --device cpu --timesteps 2
```

`--timesteps 2` is only a wiring smoke schedule. Its training report flags the
high terminal signal, and sampling from it emits a warning. The default
`--timesteps 1000` is the intended schedule for non-smoke runs.

Sample from that checkpoint:

```sh
uv run --extra train python -m exposure.pilot_train sample --checkpoint outputs/pilot-smoke-20261101-01/pilot.pt --rule red --output outputs/pilot-smoke-20261101-01/red.png --seed 0 --device cpu --guidance 1.0
```

The two-step checkpoint also warns during sampling. This command exercises
checkpoint loading and image writing; it is not evidence that the model learned.

## Exploratory GPU run

This has **not** been run yet. After the CUDA environment is ready and the
repository transfer is authorized, the same commands can use `--device cuda`:

```sh
uv run --extra train python -m exposure.pilot_train train --rules examples/exposure_rules.json --weights examples/uniform_weights.json --output outputs/pilot-run-20261101-01/pilot.pt --steps 100 --batch-size 2 --seed 0 --device cuda
```

Defaults include `--timesteps 1000`, `--width 16`, `--lr 1e-4`, `--dropout 0.1`,
`--ema-decay 0.999`, and `--max-attempts 10000`. Requesting `cuda` when PyTorch
cannot access CUDA is an error; the command does not silently fall back to CPU.

Sample with:

```sh
uv run --extra train python -m exposure.pilot_train sample --checkpoint outputs/pilot-run-20261101-01/pilot.pt --rule red --output outputs/pilot-run-20261101-01/red.png --seed 0 --device cuda --guidance 1.0
```

`--guidance` is available, but no controlled guidance sweep has been validated.
Choose new output paths for every run.

## Provenance, counts, memory, and timing

The checkpoint records `schema_version`, the configuration, rule definitions,
model, EMA, optimizer state, and the training report. The report includes:

- `device` and `last_loss` (the final batch MSE, not a validation metric);
- `elapsed_seconds`, which includes online rendering and has no warmup;
- `terminal_alpha_bar` and `prior_approximation_warning` (true when the terminal
  signal is above `0.01`);
- `peak_cuda_allocated_bytes` and `peak_cuda_reserved_bytes` (null on CPU);
- `condition_counts`, `realized_rule_counts_by_condition`, `sampling_attempts`,
  and normalized `weights`;
- a note restating that this is exploratory and makes no Stage 0 or model
  performance claim.

Sampling writes only a PNG. It does not emit scene annotations, metrics, or a
validated evaluation.

## Evidence and omissions

A separate CPU-only PyTorch 2.9.1 environment passed the real CPU unit tests and
a one-step train/checkpoint/sample CLI smoke. These are mechanics only, not
evidence of learning.

The pilot still has no resume support, reference model, validated model outputs,
controlled guidance sweep, or human audit. It does not establish GPU fit,
training quality, or any Stage 0 gate.

On the remote laptop, `uv` is installed in an isolated venv and the CUDA PyTorch
download is still in progress. GPU fit remains unverified. The repository has
not been transferred; transfer awaits explicit authorization.

See the [research and hardware plan](research-and-hardware-plan.md),
[Stage 0 status](stage0-status.md), and the
[human checklist](human-checklist.md).
