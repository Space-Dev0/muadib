# Overnight PilotUNet continuation

On the Windows GPU laptop, task `MuadibPilotUNetOvernight81390` was started
from the verified 30,000-update low/balanced checkpoint. It targets 150,000
total updates (120,000 additional) with batch 4 and FP32, preserving the
source optimizer and RNG state. It writes checkpoints every 10,000 updates
from 40,000 to 150,000 under
`%USERPROFILE%\muadib\mosaic-gpu-81300\outputs\mosaic-pilotunet-overnight-81390`.
This is one-cell feasibility training, not a MOSAIC replication or Stage 0 pass.
The trainer did not enable strict deterministic GPU algorithms, so bitwise
replay is not established despite restoring saved state.

The one-off task has no recurring or logon trigger. It was verified running
after SSH disconnected; a separate 15-second scheduled-task smoke test also
completed after SSH closed. `scripts/run-mosaic-pilotunet-overnight.ps1` checks
the parent checkpoint hash, refuses to overwrite an existing output, requests
Windows to prevent idle sleep while it runs, and releases that request afterward.
It cannot prevent explicit sleep, lid-close sleep, shutdown, or a Windows
restart. Keep the laptop powered and awake. If interrupted, preserve the run
directory, verify the latest complete checkpoint, and resume into a new output
directory; no automatic restart has been configured. A checkpoint may be
partially written if interrupted, so validate it before using it.

Check from PowerShell on the laptop:

```powershell
Get-ScheduledTask MuadibPilotUNetOvernight81390 | Select-Object TaskName, State
Get-ChildItem "$env:USERPROFILE\muadib\mosaic-gpu-81300\outputs\mosaic-pilotunet-overnight-81390" -Filter 'warmstart-*.pt'
Get-Content "$env:USERPROFILE\muadib\mosaic-pilotunet-overnight-81390.log" -Tail 30
```

At launch, the task was Running, GPU utilization was 37%, and its provenance
identified the expected 30k source hash, batch 4, 120,000 new updates and all
12 requested checkpoint steps. No first new checkpoint was present at that
point, so this records a successful launch, not completion. The previous
10,000-update segment took about 824 seconds on the laptop; timing may vary.
For the performance screen and limitations see [GPU throughput](gpu-throughput-benchmark.md).
