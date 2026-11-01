# Exploratory pilot training

**Preliminary and non-confirmatory. Stage 0 has not passed.** This is an
instrument-development pilot, not a research result. A short project run has
completed on CUDA, and no model output has been validated.

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
`uv` is inside the project virtual environment rather than on PATH. At the
user's request, the GitHub repository was made public and cloned into the
`source` subdirectory of the project folder; the existing `.venv` remains in
its parent. From the clone,
run this in PowerShell to preserve and target that environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "..\.venv"
..\.venv\Scripts\python.exe -m uv sync --locked --inexact --extra train
```

Then invoke the pilot with `..\.venv\Scripts\python.exe -m exposure.pilot_train`
and the arguments below. Locked `uv sync` and 123 Windows tests passed.

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

The CUDA environment is installed and reports CUDA as available. The repository
is cloned on the laptop. A short project run completed with `--device cuda`:

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
Choose new output paths for every run. On the laptop, invoke the same arguments
with `..\.venv\Scripts\python.exe -m exposure.pilot_train`.

## Resume completed segments

New version 2 checkpoints preserve optimizer, EMA, random states, and exposure
counts. Continue a completed segment with a new output path:

```sh
uv run --extra train python -m exposure.pilot_train resume --checkpoint outputs/pilot-run-20261101-01/pilot.pt --steps 2000 --output outputs/pilot-run-20261101-01/pilot-next.pt
```

`--steps` adds updates to the saved count. Resume requires the same device type,
visible CUDA layout, Python/PyTorch/NumPy/Pillow versions, and source hashes for
the pilot, renderer, rules, and measurement code. The implementation rejects
incompatible checkpoints; it does not promise bitwise identity across hardware
or drivers. Save completed segments regularly: checkpoints are not automatic
during a segment. Older version 1 checkpoints, including the 100- and 2,000-step
GPU explorations below, can be sampled but cannot be resumed because they lack
the required random state.

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

The pilot still has no reference model, validated model outputs,
controlled guidance sweep, or human audit. Project pilot fit on the GPU,
training quality, and any Stage 0 gate remain unestablished.

The 100-step CUDA run used four example rules, batch size 2, width 16, 1,000
diffusion timesteps, seed 43, and code revision `7d9dfe2`. Its training time was
10.997 seconds including rendering, and peak CUDA memory was 43,783,680 bytes
allocated and 65,011,712 bytes reserved. Its checkpoint and sample are in the
laptop's ignored `outputs/gpu-pilot-i6qbsa_j/` directory. The sample was visually
noise; it is not suitable for human review or evidence of learning. Longer-run
memory, quality, and the research compute budget remain unvalidated.

A second exploratory run used the same four rules and model settings, seed 44,
and 2,000 training steps (4,000 scenes). Training took 167.536 seconds
including rendering, with 43,783,680 bytes peak allocated and 65,011,712 bytes
reserved. The sampled image was still visually noise. The checkpoint and sample
are in the laptop's ignored `outputs/gpu-pilot-2000-_9x17fxs/` directory. This
short run does not establish model quality or provide an audit image set.

See the [research and hardware plan](research-and-hardware-plan.md),
[Stage 0 status](stage0-status.md), and the
[human checklist](human-checklist.md).
