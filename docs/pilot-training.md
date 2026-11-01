# Exploratory pilot training

**Preliminary and non-confirmatory. Stage 0 has not passed.** This is an
instrument-development pilot, not a research result. No project pilot model has
been trained on CUDA, and no model output has been validated.

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

The prepared Windows laptop installation is complete. Its isolated virtual
environment has `uv` 0.13.6, PyTorch 2.9.1+cu128 with CUDA 12.8, NumPy 2.5.4,
Pillow 13.0.0, and SciPy 1.18.1, and `torch.cuda.is_available()` returns true.
`uv` is inside the project virtual environment rather than on PATH. The
repository has not been transferred, so the project commands have not been run
there. Once it is present, run this in PowerShell to preserve `uv`:

```powershell
.venv\Scripts\python.exe -m uv sync --inexact --extra train
```

Then invoke the pilot with `.venv\Scripts\python.exe -m exposure.pilot_train`
and the arguments below.

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

The CUDA environment is installed and reports CUDA as available, but no project
pilot training has been run yet. After the repository transfer is authorized,
the same commands can use `--device cuda`:

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

CPU pilot tests and the one-step train/checkpoint/sample mechanical sample
passed in a separate CPU PyTorch 2.9.1 environment. These are mechanics only,
not evidence of learning.

A generic CUDA smoke also completed on the laptop. A two-layer 64×64 synthetic
convolution model ran two warmup steps and three timed CUDA
forward/backward/AdamW steps. It reported a finite loss, a measured time of
0.0695 seconds for the three timed steps, 1,804,288 bytes peak allocated, and
4,194,304 bytes peak reserved. This is a generic CUDA smoke only. It is not
project pilot fit, learning evidence, Stage 0 passage, or a study.

The pilot still has no resume support, reference model, validated model outputs,
controlled guidance sweep, or human audit. Project pilot fit on the GPU,
training quality, and any Stage 0 gate remain unestablished.

The repository has not been transferred to the laptop, and no project model
training has run on the GPU.

See the [research and hardware plan](research-and-hardware-plan.md),
[Stage 0 status](stage0-status.md), and the
[human checklist](human-checklist.md).
