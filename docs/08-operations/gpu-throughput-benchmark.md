# PilotUNet GPU throughput screen

On the Windows laptop, a short isolated synthetic-input benchmark screened
training speed before the overnight run. It did not load or modify the research
checkpoint, write training artifacts, or evaluate image quality. Hardware was
an NVIDIA GeForce RTX 3050 6GB Laptop GPU (6,144 MiB), Windows driver 610.91,
PyTorch 2.9.1+cu128. The tested model was `PilotUNet(width=16, timesteps=1000)`
with AdamW, EMA updates, 64×64 random inputs already resident on the GPU,
velocity loss, conditional dropout and per-update loss synchronization. This
approximates the computational path but is not a complete benchmark of the
actual dataset or overnight trainer.

The second short run measured five warmup and 20 timed updates per setting,
with CUDA synchronization at the start and end of each timed region. Times
are a rough screen, not repeated-run confidence intervals; thermal state,
Windows scheduling, GPU power limits and other processes can change them.
Peak values below are PyTorch allocated memory, not total GPU usage.

| Batch | Mode | Updates/s | Images/s | Peak allocated MiB |
|---:|---|---:|---:|---:|
| 4 | FP32 | 13.55 | 54.2 | 70.2 |
| 4 | TF32 permitted | 13.33 | 53.3 | 73.0 |
| 4 | FP32, cuDNN benchmark | 13.65 | 54.6 | 70.2 |
| 4 | AMP FP16 with GradScaler | 10.30 | 41.2 | 68.0 |
| 4 | AMP BF16 | 10.65 | 42.6 | 68.2 |
| 8 | FP32 | 13.93 | 111.4 | 107.3 |
| 8 | AMP FP16 with GradScaler | 9.87 | 79.0 | 103.3 |
| 16 | AMP FP16 with GradScaler | 10.16 | 162.5 | 171.6 |

All measured losses were finite over these brief runs. That does not establish
numerical equivalence, stable long training, or model quality. The first run
measured batch-4 FP32 at 12.08 updates/s and TF32 at 13.20 updates/s, while the
second reversed their order; these short measurements do not establish a TF32
speedup. Mixed precision was slower at batch 4 in the second run. Batch 8
increased images per second but changes the gradient batch and optimization
trajectory, and the current checkpoint continuation enforces the original
batch size. GPU utilization percentage alone does not establish a bottleneck.

The trainer converts loss and checks finite values on the host each update;
`PilotUNet.forward` also converts a device boolean to a Python boolean. These
are possible synchronization overheads, not a demonstrated bottleneck.
Profiling and a numerically checked isolated experiment are needed before
changing the training loop or backend flags. Preserve the existing settings
for a comparable resumed run.

The exact benchmark source was copied from the laptop probe to
`scripts/benchmark-pilotunet-gpu.py`. To reproduce on the staged Windows source
with its CUDA virtual environment, copy that script into the staged root, then
run **after** any active training has stopped:

```powershell
Set-Location "$env:USERPROFILE\muadib\mosaic-gpu-81300"
$env:PYTHONPATH = (Get-Location).Path
& "$env:USERPROFILE\muadib\.venv\Scripts\python.exe" -u .\benchmark-pilotunet-gpu.py
```

The exact runs used temporary staged files `gpu-benchmark-81431.py` and
`gpu-benchmark-81432.py`; the latter only changed JSON reporting to one line
per setting. The benchmark creates models and performs temporary optimizer
updates in memory. Do not run it concurrently with the overnight experiment.
