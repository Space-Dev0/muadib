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
- Strict deterministic training is opt-in via `--deterministic`. When enabled,
  it configures process-wide PyTorch determinism flags and cuBLAS workspace
  settings; unsupported operations raise runtime errors. Checkpoints use
  schema version 3 and record determinism flags. Older v1/v2 checkpoints can
  still be sampled in default mode but cannot be resumed.

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
and the arguments below. Locked `uv sync` and all 127 updated Windows tests passed.

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

To opt into strict deterministic execution on CUDA, pass `--deterministic`:

```sh
uv run --extra train python -m exposure.pilot_train train --rules examples/exposure_rules.json --weights examples/uniform_weights.json --output outputs/pilot-run-20261101-01/pilot.pt --steps 100 --batch-size 2 --seed 0 --device cuda --deterministic
```

When `--deterministic` is enabled:

- `CUBLAS_WORKSPACE_CONFIG` is set to `:4096:8` (or `:16:8` if already present in
  the environment) before CUDA is initialized. If CUDA has already been
  initialized in the process before this configuration, training raises an error
  requiring a fresh process.
- Process-wide PyTorch flags are enabled: `torch.use_deterministic_algorithms(True, warn_only=False)`,
  `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`,
  and TF32 math is disabled for both cuDNN and matrix multiplications
  (`torch.backends.cudnn.allow_tf32 = False`, `torch.backends.cuda.matmul.allow_tf32 = False`).
- Operations without deterministic implementations raise runtime errors.
- Strict determinism does not guarantee bitwise reproducibility across different
  hardware architectures, CUDA versions, or driver versions. The integrated
  `--deterministic` CUDA flag has now been tested on the RTX 3050: two independent
  two-step runs matched model weights, a four-step run matched a 2+2 resumed run
  for model, EMA, optimizer, Python/CPU/CUDA RNG and exposure counts, and two
  same-seed sample PNGs matched byte-for-byte. This is a small two-timestep wiring
  test, not evidence of research quality or determinism on other hardware.

Sample with:

```sh
uv run --extra train python -m exposure.pilot_train sample --checkpoint outputs/pilot-run-20261101-01/pilot.pt --rule red --output outputs/pilot-run-20261101-01/red.png --seed 0 --device cuda --guidance 1.0
```

`--guidance` is available, but no controlled guidance sweep has been validated.
Choose new output paths for every run. On the laptop, invoke the same arguments
with `..\.venv\Scripts\python.exe -m exposure.pilot_train`.

## Resume completed segments

New version 3 checkpoints preserve optimizer, EMA, random states, determinism
configuration, and exposure counts. Continue a completed segment with a new output path:

```sh
uv run --extra train python -m exposure.pilot_train resume --checkpoint outputs/pilot-run-20261101-01/pilot.pt --steps 2000 --output outputs/pilot-run-20261101-01/pilot-next.pt
```

`--steps` adds updates to the saved count. Resume requires checkpoint schema
version 3, the same device type, visible CUDA layout, matching determinism
mode (inherited automatically), Python/PyTorch/NumPy/Pillow versions, and source
hashes for the pilot, renderer, rules, and measurement code. The implementation
rejects incompatible checkpoints; it does not promise bitwise identity across
hardware or drivers. Save completed segments regularly: checkpoints are not
automatic during a segment.

### Compatibility with older checkpoints

- **Schema version 3**: Supports both resume and sampling. When sampling from a
  version 3 checkpoint, the sampling command automatically applies the
  determinism mode saved in the checkpoint's configuration.
- **Older v1 and v2 checkpoints**: Can still be sampled in the default
  non-strict mode. However, they cannot be resumed (resuming strictly requires
  schema version 3 and complete random stream state; attempting to resume an
  older checkpoint raises a `ValueError`).

## Provenance, counts, memory, and timing

The checkpoint records `schema_version` (now schema version 3), the
configuration (including `deterministic`), rule definitions, model, EMA,
optimizer state, the training report, and the `rng` state. The runtime state in
`rng["runtime"]` captures PyTorch determinism flags (`deterministic_algorithms`,
`cudnn_deterministic`, `cudnn_benchmark`, `cudnn_allow_tf32`,
`matmul_allow_tf32`, and `cublas_workspace_config`). The report includes:

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

## One-step denoising probe

`exposure.denoise_probe` compares raw weights, EMA weights, and a zero-noise
baseline on the same seeded CPU renders and noise draws. It reports noise MSE
and reconstructed clean-image MSE at each requested timestep. For example:

```sh
uv run --extra train python -m exposure.denoise_probe --checkpoint outputs/deterministic-pilot-5549zx7k/step-8000.pt --rule red --samples 8 --timesteps 20 100 500 --seed 370 --output outputs/denoise-probe-8000-red.json
```

The output path must be new. On the laptop, use the parent virtual-environment
Python as described above. Comparing the 4,000- and 8,000-step checkpoints on
the same eight renders and noise draws gave EMA noise MSEs of 0.2644 versus
0.1378 at timestep 20, 0.0804 versus 0.0510 at 100, and 0.0323 versus 0.0162
at 500; the zero predictor was approximately 1 at each timestep. This small
one-step check does not validate full generation or establish why samples are
noisy.

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
controlled guidance sweep, or human audit. Training quality and the Stage 0
gates remain unestablished. The deterministic CUDA check above was a small
two-timestep wiring test, not evidence of research quality.

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

A 4,000-step deterministic project pilot was measured across two 2,000-update
segments with checkpoint resume, using batch size 2, width 16, 1,000 diffusion
timesteps, seed 45, and code revision `4287d88`. Cumulative recorded training
time was 409.956 seconds, with peak allocated memory of 96,212,480 bytes and
reserved memory of 111,149,056 bytes. The single sampled red image was still
mostly noise, so it is not suitable for human audit or evidence of research
quality; Stage 0 remains unpassed.

An 8,000-step deterministic pilot was measured at code revision `4287d88`:
training continued the existing seed-45, batch-size-2, width-16, 1,000-timestep
run in two additional 2,000-step checkpointed segments. Cumulative recorded
training time was 802.497 seconds, with peak allocated memory of 96,212,480
bytes and reserved memory of 111,149,056 bytes. Four example conditions were
sampled once each using seeds 5, 6, 7, 8; visual inspection found noise or
near-blank images with colored fragments, none clearly showing the intended
three-object scene. This is only four inspected samples, not a formal quality
score, and they are not ready for human audit. Checkpoints and samples are in
ignored `outputs/deterministic-pilot-5549zx7k` on the laptop. Stage 0 remains
explicitly unpassed.

For the `red` condition and seed 5, separate samples from the raw and EMA
weights were both visually noisy. This single comparison does not establish
whether EMA affects quality in general.

See the [research and hardware plan](research-and-hardware-plan.md),
[Stage 0 status](stage0-status.md), and the
[human checklist](human-checklist.md).
