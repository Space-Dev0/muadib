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

## Single-scene overfit diagnostic

`exposure.static_scene_probe` trains on one fixed render to investigate the
training and sampling pipeline. It is separate from the online experiment:

```sh
uv run --extra train python -m exposure.static_scene_probe --rules examples/exposure_rules.json --rule red --output-dir outputs/static-scene-example --steps 4000 --device cuda --deterministic
```

The output directory must be new. It contains `target.png`, `sample.png`, a
diagnostic `checkpoint.pt`, and `report.json` with the scene and seeds. The
checkpoint is intentionally not resumable as an online run. The renderer
override is process-wide during this command, so run the probe in its own
process. A single target and sample cannot establish generalization or a
Stage 0 result.

A deterministic diagnostic run on the RTX 3050 executed 4,000 updates with
batch size 2, width 16, 1,000 diffusion timesteps, train seed 46, target scene
seed 77, sample seed 5, and code revision `35500f7`. Training finished in
365.534 seconds, with peak allocated CUDA memory of 96,207,872 bytes and peak
reserved memory of 111,149,056 bytes. The fixed target image clearly contained
three objects, while the one sampled image looked mostly dark noise (image MSE
on `[0, 1]` was 0.8017). Artifacts are in ignored
`outputs/static-scene-769aef5ec6` on the laptop. This is one diagnostic sample
on a single fixed scene; it does not establish a model bug, training failure in
general, or Stage 0 passage.

## Truncated reconstruction diagnostic

`exposure.reconstruction_probe` runs an exploratory truncated-reconstruction
diagnostic on CPU comparing raw weights and EMA weights on an explicit 64×64
target image. It loads either an online pilot checkpoint or a static-scene
diagnostic checkpoint:

```sh
uv run --extra train python -m exposure.reconstruction_probe --checkpoint outputs/static-scene-example/checkpoint.pt --target outputs/static-scene-example/target.png --rule red --timesteps 20 100 500 --seeds 0 1 --output outputs/reconstruction-probe-example.json
```

The output path must be new; the CLI refuses to overwrite existing files,
the checkpoint, or the target image. Omitting `--output` prints the JSON report
to stdout. For each selected start timestep, the diagnostic forward-noises the
target using the checkpoint schedule's actual $\bar{\alpha}$ and a seeded
generator, then reverses from that timestep to zero using the existing DDPM
$\epsilon$-mean and posterior variance equations.

Key caveats and operational details:

- **Schedule preservation**: The checkpoint's full 1,000-step schedule is kept
  unchanged; the schedule is never shortened or rescaled to the truncated range.
- **Matched noise**: Raw and EMA evaluations use identical matched starting
  noise and identical matched transition noise at each reverse step.
- **Reporting**: The diagnostic reports reconstruction MSE and baseline input
  MSE per seed for each requested timestep, as well as summary averages across
  seeds. It outputs numerical measurements only and never fabricates human
  labels or applies arbitrary pass/fail thresholds.
- **Instrument check only**: This separate reverse loop is an instrument check
  and is not proof of sampler correctness, population quality, or Stage 0
  passage.
- **Source fingerprint untouched**: `exposure/pilot_train.py` is not modified,
  preserving source hashes so existing GPU checkpoints remain resumable.

For the 4,000-update fixed-scene checkpoint described above, EMA reconstruction
MSE on the `[-1, 1]` scale at timesteps 20, 100 and 200 was 0.0022, 0.0060 and
0.0105, versus noisy-input MSE 0.0062, 0.1069 and 0.3750, averaged over ten
seeds. At timesteps 500, 750 and 999, EMA reconstruction MSE was 0.1185,
0.3760 and 9.4313, versus noisy-input MSE 1.4104, 1.8292 and 1.9249,
averaged over three seeds. The highest-noise reconstruction worsened in this
setup; this single target and separate reverse loop cannot isolate a cause or
establish general model quality. Local reports are in ignored files
`outputs/reconstruction-static-4000.json` and
`outputs/reconstruction-static-4000-high.json`.

## One-step high-t baseline diagnostic

`exposure.high_t_baseline` is a one-step CPU diagnostic for one fixed target. It
compares raw and EMA noise predictions with trivial copy-noisy-input,
zero-noise, and exact-oracle predictors. For each requested timestep it
forward-noises the target with the checkpoint schedule's actual
$\bar{\alpha}$ and matched seeded noise, then scores noise MSE and reconstructed
clean-image MSE.

Run the recorded diagnostic on the 4,000-update fixed-scene checkpoint:

```sh
uv run --extra train python -m exposure.high_t_baseline --checkpoint outputs/static-scene-769aef5ec6/checkpoint.pt --target outputs/static-scene-769aef5ec6/target.png --rule red --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --timesteps 900 950 990 999 --output outputs/high-t-baseline-static-4000.json
```

The output path must be new. On the laptop, use
`..\.venv\Scripts\python.exe -m exposure.high_t_baseline` with the same
arguments.

Mean one-step noise MSE over seeds 0–15:

| Predictor | t=900 | t=950 | t=990 | t=999 |
| --- | --- | --- | --- | --- |
| Raw | 0.0205755 | 0.0211872 | 0.0199738 | 0.0198872 |
| EMA | 0.0279802 | 0.0283686 | 0.0278234 | 0.0273750 |
| Copy noisy input | 0.0002520 | 0.0000989 | 0.0000451 | 0.0000376 |

The reported zero-predictor noise MSE was about 0.995819 and the exact oracle
was 0. The copy predictor returns the noisy input as its noise estimate, so its
low error at these high timesteps is expected and is not a model result. Raw
noise MSE was lower than EMA at all four timesteps in this check; one target and
one checkpoint cannot establish a general raw/EMA difference.

This is one training target and one checkpoint. It is not a full reverse
sampling run, does not validate generated images, and is not a generalization or
Stage 0 result. The report also records unclipped and clipped `x0` MSE and the
fraction of reconstructed pixels outside `[-1, 1]`; the outside fraction uses a
`1e-4` tolerance for floating-point error. The JSON is in ignored
`outputs/high-t-baseline-static-4000.json`.

## Paired high-t fine-tuning diagnostic

`exposure.high_t_finetune` runs two matched fixed-scene arms for 500 additional
updates each, from the same static-scene checkpoint and target: an unweighted
control and a high-timestep-weighted arm. Both use batch size 2, training seed
46, the full 1,000-step schedule, and identical sampled timesteps; the evaluated
seeds were 0–7. The weighted arm uses multiplier 5 at timesteps 900 and above,
normalized by the expected weight 1.4. Both arms realized a high-timestep
fraction of 0.103. Training used the RTX 3050 at code revision `4c8c798`;
one-step errors were evaluated on CPU.

```sh
uv run --extra train python -m exposure.high_t_finetune --checkpoint outputs/static-scene-769aef5ec6/checkpoint.pt --target outputs/static-scene-769aef5ec6/target.png --output-dir outputs/high-t-paired-4c8c798 --rule red --steps 500 --batch-size 2 --seed 46 --device cuda --deterministic --high-t-min 900 --high-t-weight 5 --eval-timesteps 20 100 500 900 990 999 --eval-seeds 0 1 2 3 4 5 6 7
```

The output directory must be new. On the laptop, use
`..\.venv\Scripts\python.exe -m exposure.high_t_finetune` with the same
arguments. The command writes `baseline.pt`, `weighted.pt`, and `report.json`.

Mean one-step noise MSE:

| Arm | t=999 raw | t=999 EMA | t=100 raw |
| --- | --- | --- | --- |
| Start | 0.019759 | 0.027238 | — |
| Control (unweighted) | 0.017361 | 0.022943 | 0.044935 |
| Weighted (5× high-t) | 0.016514 | 0.022615 | 0.051213 |
| Copy noisy input at t=999 | 0.000038 | — | — |

At t=999, weighted training reduced raw and EMA noise MSE relative to control;
at t=100, weighted raw noise MSE was higher than control. Training took 51.57
seconds for control and 50.41 seconds for weighted, with peak allocated CUDA
memory of 96,152,064 bytes for both arms. The copy-noisy-input baseline remains
a trivial reference, not a model result.

This is one fixed scene, one checkpoint, and one training seed. It reports
one-step errors only and does not establish full-sampling behavior,
generalization, human audit accuracy, or any Stage 0 result. Remote artifacts
are in the ignored `outputs/high-t-paired-4c8c798` directory.

One matched full sample from each arm, using seed 5 and EMA weights, appeared
mostly dark noise on visual inspection. These two samples are not a population
quality estimate and are not ready for the human checker audit.

## Evidence and omissions

The original online velocity run reached 100,000 updates (23637.6 cumulative
seconds, 672003 rendering attempts). Requested counts were 50001 red, 50442
square, 49717 red-left-of-square and 49840 blue-above-circle. A seed-5 red
sample showed three recognizable shapes including red; a seed-6 relational
sample showed only one clearly visible blue shape. This inspection of two
samples is not an accuracy estimate. The checkpoint and samples remain in
ignored laptop paths beginning `outputs/velocity-online-v1-100000`.

The current checker rejected both exploratory samples: the red sample for
unsupported object size or occlusion, and the relational sample because it
found one component instead of three. These two rejections illustrate checker
coverage limitations on this checkpoint; they do not establish its error rate
or substitute for the independent human audit.

An additional exploratory preview generated four images per pilot condition
(`red`, `square`, `red_left_of_square`, `blue_above_circle`) at seeds
101–104 and guidance 1, using the original checkpoint-compatible code. Visual
inspection found recognizable shapes in many images, but some showed fewer
than three visible objects and relational satisfaction was not consistently
clear. This 16-image convenience preview is not a blinded audit, an accuracy
estimate or evidence of Stage 0 passage. Images and manifest are in ignored
laptop directory `outputs/velocity-preview-100000-20261108/`.

### Exploratory optimizer comparison

At revision `61ae41a`, fresh AdamW and momentum-SGD models trained for 1,000
updates each on one fixed target with matched initialization, timesteps and
noise, batch size 2. AdamW used learning rate 0.0001; SGD used 0.01 and
momentum 0.9. Both used weight decay 0.01, decoupled for AdamW and coupled
for SGD. Their seed-5 and seed-6 sample MSE on `[0,1]` was 0.16130/0.16723
for AdamW and 0.08056/0.08362 for SGD, versus 0.03874 for the constant
target-mean image. One sample per arm appeared noisy. These learning rates
were not tuned, the objectives' optimization details differ, and this one
scene and training seed cannot establish a general optimizer winner. Laptop
artifacts are in ignored `outputs/optimizer-probe-61ae41a/`.

The checkpointed online AdamW pilot also reached 32,000 updates, with
3,395.2 cumulative seconds and 215,296 rendering attempts. Requested counts
were 16,017 red, 16,109 square, 15,907 red-left-of-square and 15,967
blue-above-circle. In two matched samples the red condition showed a red
shape; the relational sample did not clearly satisfy the requested relation.
These are visual observations, not an accuracy estimate. Artifacts are in
ignored laptop paths beginning `outputs/velocity-online-v1-32000-79c331f`.

### Conditioning-use diagnostic

`exposure.velocity_condition_probe` evaluates the EMA velocity model on fresh
annotated renders using the same image, noise and timestep under the requested
rule, a rule verified false for that scene, and null conditioning. Positive
false-minus-correct loss means the requested rule predicted velocity better
for that comparison. Cases with no available false rule are excluded and
reported. This teacher-forced diagnostic does not evaluate generated-image
accuracy or establish a causal explanation.

On the 12,000-update checkpoint, 16 scene seeds per requested rule and five
timesteps produced 310 paired rows with two scenes excluded. Pooled mean
false-minus-correct velocity MSE was 0.000526, 0.000178, 0.000479, 0.000504
and 0.000464 at timesteps 20, 100, 500, 900 and 999 respectively. The null
comparison was mixed at the earliest timestep. These small descriptive
differences have no uncertainty intervals; rules, scenes and timesteps are not
independent replicates. Different seeds alone do not guarantee disjoint scenes.
The local ignored report is `outputs/velocity-condition-12000-cpu.json`.

The same scene seeds, noise seed and timesteps evaluated the 20,000-update
checkpoint on CPU. It also retained 310 paired rows and excluded two scenes.
Pooled mean false-minus-correct velocity MSE at timesteps 20, 100, 500, 900
and 999 was 0.000583, 0.000117, 0.000844, 0.000468 and 0.000484. The
corresponding 12,000-update values were 0.000526, 0.000178, 0.000479,
0.000504 and 0.000464. These are descriptive comparisons on the same small
set, with no uncertainty intervals; they do not establish that generated
conditions have improved. The local ignored report is
`outputs/velocity-condition-20000-cpu.json`.

```sh
uv run --extra train python -m exposure.velocity_condition_probe --checkpoint outputs/velocity-online-v1-12000-79c331f.pt --output outputs/velocity-condition-new.json --scene-seeds 76000 76001 76002 76003 76004 76005 76006 76007 76008 76009 76010 76011 76012 76013 76014 76015 --noise-seed 91234 --timesteps 20 100 500 900 999 --device cpu
```

### Matched velocity-prediction fixed-scene experiment

At revision `06a95ff`, the RTX 3050 trained matched epsilon- and velocity-
prediction models for 4,000 updates each on one fixed scene, batch size 2,
with identical initialization, random streams and 1,000-step schedule. At
timestep 999, EMA noise MSE across four evaluation seeds was 0.026877 for
epsilon and 0.000005 for velocity; at timestep 20 it was 0.197213 and
0.415274, respectively. Objective losses are not directly comparable.

For sample seeds 5 and 6, image MSE on `[0, 1]` was 0.30331/0.40558 for
epsilon and 0.04987/0.05709 for velocity; a constant target-mean image has MSE
0.03874. The velocity images show colored shapes but do not faithfully reproduce
the three-object target. This is one training scene, one training seed and two
samples, not evidence of generalization, valid human audit images or Stage 0
passage. Training took 365.6 and 354.7 seconds, with 96,165,888 and 99,489,280
bytes peak allocated, respectively. Laptop artifacts are in ignored
`outputs/velocity-probe-06a95ff/`.

```sh
uv run --extra train python -m exposure.velocity_probe --rules examples/exposure_rules.json --target outputs/static-scene-769aef5ec6/target.png --output outputs/velocity-probe-new --rule red --steps 4000 --batch-size 2 --seed 46 --init-seed 46 --width 16 --timesteps 1000 --device cuda --deterministic --eval-times 20 100 500 900 999 --eval-seeds 0 1 2 3 --sample-seeds 5 6
```

The velocity diagnostic checkpoint requires its custom prediction conversion
and is explicitly rejected by standard pilot sampling and diagnostics.

At revision `edf3dc3`, an online velocity pilot drew fresh conditioned scenes
for 4,000 updates with batch size 2, seed 48, four example conditions and the
full 1,000-step schedule. The RTX 3050 run took 401.8 seconds, with 96,114,176
bytes peak allocated and 111,149,056 bytes reserved. Requested condition
counts were 1,994 red, 1,975 square, 1,990 red-left-of-square and 2,041
blue-above-circle. The report also records realized rule counts and 26,987
rendering attempts. A red sample (seed 5) and blue-above-circle sample (seed 6)
contained colored blobs but no clearly identifiable shapes on inspection.
These are two exploratory samples and do not measure population quality or
pass Stage 0. Artifacts are in ignored laptop paths beginning
`outputs/velocity-online-4000-edf3dc3`.

New online velocity runs use schema 1 checkpoints and can continue with
`python -m exposure.velocity_online resume --checkpoint OLD.pt --steps N --output NEW.pt`
under the same recorded code, dependency versions, device layout and
determinism settings. Schema 0 checkpoints, including the 4,000-update run
above, remain sampleable but lack optimizer and random state and cannot be
resumed. CPU tests verify exact uninterrupted versus resumed replay; CUDA replay
was also checked on the target laptop as described below.

At revision `79c331f`, a deterministic CUDA replay of four uninterrupted
updates versus two plus two resumed updates matched model, EMA, optimizer,
Python and Torch random states, requested counts, realized rule counts and
rendering attempts exactly. This small check does not guarantee replay across
different hardware or runtimes.

A fresh schema 1 online pilot was trained in two 4,000-update segments to 8,000
updates, batch size 2 and seed 48. The RTX 3050 recorded 805.6 cumulative
seconds, 96,114,176 bytes peak allocated and 111,149,056 bytes reserved.
Requested counts were 3,970 red, 3,990 square, 3,946 red-left-of-square and
4,094 blue-above-circle, with 53,578 rendering attempts. One red sample (seed
5) and one relational sample (seed 6) showed distinct colored blobs but not
clearly identifiable shapes or a reliable requested relation. These two images
are exploratory and not ready for the human checker audit. Checkpoints,
reports and samples are in ignored laptop paths beginning
`outputs/velocity-online-v1-`.

The same run continued to 12,000 updates at revision `79c331f`. Cumulative
training time was 1,207.0 seconds, with the same 96,114,176-byte allocated and
111,149,056-byte reserved memory peaks. Requested counts were 5,962 red,
5,997 square, 6,014 red-left-of-square and 6,027 blue-above-circle, with
80,565 rendering attempts. The red sample (seed 5) and relational sample
(seed 6) appeared somewhat cleaner but did not clearly satisfy their requested
conditions: the first had no obvious red object and the second did not show a
clear blue-above-circle relation. These two visual observations are not a
population evaluation or a Stage 0 result. The resumable checkpoint and
artifacts are in ignored laptop paths beginning
`outputs/velocity-online-v1-12000-79c331f`.

The checkpointed run subsequently reached 16,000 updates in 1,605.2 cumulative
seconds. Requested counts were 7,994 red, 8,032 square, 7,983 red-left-of-square
and 7,991 blue-above-circle, with 107,150 rendering attempts. The two matched
red and relational samples remained blurry and did not clearly satisfy their
requested conditions. These two images are not a population evaluation or a
Stage 0 result. Artifacts are in ignored laptop paths beginning
`outputs/velocity-online-v1-16000-79c331f`.

The same checkpointed run reached 20,000 updates in 2,096.4 cumulative seconds.
Requested counts were 9,994 red, 10,020 square, 9,967 red-left-of-square and
10,019 blue-above-circle, with 133,972 rendering attempts. The matched red
sample (seed 5) visibly contained a red object; the relational sample (seed 6)
did not clearly show the requested relation. This is a visual inspection of two
images, not a quality estimate or Stage 0 result. The checkpoint and artifacts
are in ignored laptop paths beginning `outputs/velocity-online-v1-20000-79c331f`.

At 24,000 updates, the same run recorded 2,493.7 cumulative seconds and
161,440 rendering attempts. Requested counts were 11,980 red, 12,061 square,
11,952 red-left-of-square and 12,007 blue-above-circle. The matched seed-5 red
sample showed a clearer red object; the seed-6 relational sample still did not
clearly satisfy blue-above-circle. This is an informal inspection of two
images, not an accuracy estimate. Checkpoint and samples are in ignored laptop
paths beginning `outputs/velocity-online-v1-24000-79c331f`.

At revision `f97c8f4`, a classifier-free guidance sweep used that checkpoint,
conditions `red` and `blue_above_circle`, seeds 5 and 6, and scales 1, 2 and 4.
Initial and transition noise were matched across scales for each condition and
seed. The red/seed-5 scale-1 sample matched the earlier saved image pixel for
pixel. Higher scales visibly changed colors and sometimes suppressed objects;
the 12 inspected images still had blurry shapes and did not consistently show
the requested condition. This is a small exploratory comparison, not a human
audit, selected guidance setting, or Stage 0 pass. Laptop images and the report
are in ignored `outputs/velocity-guidance-f97c8f4/`.

```sh
uv run --extra train python -m exposure.velocity_guidance --checkpoint outputs/velocity-online-v1-12000-79c331f.pt --output outputs/velocity-guidance-new --rules red blue_above_circle --seeds 5 6 --scales 1 2 4 --device cuda
```

The standard pilot tools now reject diagnostic checkpoints marked as requiring
a custom predictor. This guard changes the source fingerprint in
`exposure/pilot_train.py`: existing version 3 online checkpoints cannot resume
at this revision. To resume them reproducibly, use the exact original code
revision and compatible recorded runtime; ordinary sampling of standard
checkpoints remains available. Do not remove the fingerprint check.

The isolated `exposure.time_embedding_probe` compares the pilot's learned
timestep lookup against a fixed sinusoidal representation on one fixed scene,
with matched non-time initialization and random streams. The two time
representations have different parameter counts. The sinusoidal checkpoint is
marked as requiring its custom architecture; standard pilot tools reject it.
The hypothesis is motivated by sparse exposure to individual learned indices
and by the sinusoidal timestep embedding in the reference
[Improved DDPM implementation](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/nn.py).
This is an experiment, not evidence that timestep embeddings caused failure.

At revision `b20489f`, the RTX 3050 ran the comparison for 4,000 updates per
arm on the same fixed scene and paired random streams, batch size 2, training
and initialization seeds 46, with 1,000 diffusion timesteps. At timesteps 20,
100, 500 and 999, EMA noise MSE across four fixed evaluation seeds was
0.1778/0.0735/0.0305/0.0271 for learned embeddings and
0.1567/0.0489/0.0283/0.0257 for sinusoidal embeddings. Two full samples per arm
(seeds 5 and 6) had `[0, 1]` pixel MSE 0.7670/0.8212 and 0.8721/0.8915,
respectively; a constant target-mean image has MSE 0.0387. One sample from
each arm looked like dark noise. This single-scene result does not demonstrate
that sinusoidal embeddings solve sampling or establish a general performance
comparison. The laptop's ignored `outputs/time-embedding-b20489f/` retains the
report, checkpoints and images. The recorded arm times (364.7 and 767.2
seconds) were affected by a laptop/network interruption and should not be used
as a hardware speed comparison.

The same four-seed one-step report shows EMA reconstructed-image MSE on the
`[-1, 1]` scale at t=20 of 0.00112 (learned) and 0.00099 (sinusoidal), versus
0.00629 for a zero-noise predictor. At t=999, the respective noise MSEs were
0.02706 and 0.02565 versus about 0.000038 for copying the noisy input. These
one-step measurements locate a high-noise weakness for this target but do not
isolate the cause of failure in a full reverse trajectory.

### Matched identity-residual fixed-scene experiment

At revision `ac70a5d`, two models were trained from identical zero-output-head
initialization on the same target with the same random streams: standard noise
prediction and `epsilon = noisy_input + residual`. Each ran 4,000 updates on
the RTX 3050, batch size 2, seeds 46, with the full 1,000-step schedule. EMA
one-step noise MSE across four evaluation seeds was 0.026877 versus 0.001895
at timestep 999 (standard versus identity-residual), but 0.197213 versus
0.452320 at timestep 20. The timestep-999 copy-input baseline was about
0.000038. Training took 372.1 and 366.6 seconds, respectively.

For sample seeds 5 and 6, identity-residual image MSE on `[0, 1]` was 0.2423
and 0.2564, versus 0.3033 and 0.4056 for standard. A constant target-mean
image has MSE 0.0387; the pure identity-epsilon sampler had 0.2724 and 0.2789.
Inspection of one sample from each trained arm showed noise. MSE is dominated
by the light background and does not measure whether objects or rules are
correct. These are two samples from one scene and one training seed, not
population evidence or Stage 0 passage. Laptop artifacts are in ignored
`outputs/identity-eps-ac70a5d/`.

```sh
uv run --extra train python -m exposure.identity_eps_finetune --rules examples/exposure_rules.json --target outputs/static-scene-769aef5ec6/target.png --output-dir outputs/identity-eps-new --rule red --steps 4000 --batch-size 2 --seed 46 --init-seed 46 --width 16 --timesteps 1000 --device cuda --deterministic --eval-timesteps 20 100 500 900 999 --eval-seeds 0 1 2 3 --sample-seeds 5 6
```

Identity-residual checkpoints use the diagnostic's own sampler; they cannot be
treated as standard noise-prediction checkpoints.

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

The same run continued to 12,000 updates in two more checkpointed segments.
Cumulative training time was 1,210.404 seconds; peak allocated and reserved
memory stayed at 96,212,480 and 111,149,056 bytes. One `red` sample and one
`blue_above_circle` sample were visually noise. This is only two inspected
samples and does not measure the full output distribution.

On eight matched `red` renders, the 12,000-step EMA one-step noise MSE was
0.1104, 0.0394, and 0.0115 at timesteps 20, 100, and 500 respectively, versus
0.1378, 0.0510, and 0.0162 at 8,000 steps. A separate eight-render check at
timestep 999 had noise MSE 0.0072 but reconstructed-image MSE about 179 on the
`[-1, 1]` scale. Small noise errors are amplified when the remaining signal is
very small; this observation does not by itself identify the cause of noisy
samples. An exact-noise oracle also reconstructed a known image through the
existing sampler, which checks its wiring without validating learned output.

A separate exploratory sampler comparison clipped the predicted clean image
to `[-1, 1]` at each reverse step. For the same checkpoint, conditions, and
seeds, its unclipped runs exactly matched the saved baselines; both clipped
samples were still visually noisy. At timestep 999, about 92% of predicted
pixels were outside the clipping range in both examples. This is only two
paired samples, so it does not settle whether clipping helps in general. The
default sampler was not changed. The local ignored files
`outputs/x0clip-12k-experiment.py` and `outputs/x0clip-12k-report.json` retain
the one-off script and measurements.

See the [research and hardware plan](research-and-hardware-plan.md),
[Stage 0 status](stage0-status.md), and the
[human checklist](human-checklist.md).
