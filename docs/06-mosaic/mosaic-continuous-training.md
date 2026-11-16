# Continuous PilotUNet training

The continuous supervisor `MuadibPilotUNetContinuous81391` waits until the
existing `MuadibPilotUNetOvernight81390` task stops, then starts successive
10,000-update segments with no configured final update count. It does not
start a second trainer while the predecessor task is running. It uses the same
low/balanced dataset, batch size 4, FP32 model and checkpointed optimizer/RNG
states. It is a single-cell feasibility model, not a MOSAIC replication or
Stage 0 pass. Strict deterministic GPU replay was not enabled.

Each segment has a new exclusive output directory under
`%USERPROFILE%\muadib\mosaic-gpu-81300\outputs\mosaic-pilotunet-continuous-81391`.
The supervisor requires a completed report, matching checkpoint SHA-256,
loadable checkpoint, expected training configuration, dataset fingerprint,
Python/PyTorch and source fingerprints before advancing. It records the last
verified checkpoint in `state.json`. If the original predecessor is interrupted
before its final report exists, it instead searches its snapshots for a
loadable checkpoint with the expected configuration; without a completed report
there is no independent expected file hash, which is explicitly recorded. It
keeps partial directories and stops on errors rather than repeatedly retrying.

The task has a Windows sign-in trigger and is configured to start again after
that user signs in following a reboot. A persistent STOP marker prevents it
from restarting after a user stop. The supervisor requests that Windows avoid
idle sleep while it runs, but explicit sleep, lid-close sleep, shutdown, sign-in
requirements, failures or lack of disk space can still interrupt training.
Checkpoint files continue to accumulate; monitor free disk space. An actual
reboot/sign-in recovery has not been exercised in this session.

## Stop

On the GPU laptop, run PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\muadib\stop-mosaic-pilotunet-continuous.ps1"
```

The command writes a persistent STOP marker, disables the sign-in task, and
stops the original predecessor if it is still active. The supervisor checks
the marker and terminates its own active process tree; an in-progress segment
may lose work and leave partial files. It does not delete checkpoints. Check
that both tasks have stopped and inspect the supervisor log:

```powershell
Get-ScheduledTask MuadibPilotUNetOvernight81390,MuadibPilotUNetContinuous81391 | Select-Object TaskName,State
Get-Content "$env:USERPROFILE\muadib\mosaic-pilotunet-continuous-81391\supervisor.log" -Tail 40
```

To intentionally restart after stopping, remove the STOP file, enable the
continuous task, and start it manually; this is a new user decision. Do not
manually delete or overwrite partial segment directories. The supervisor
validates `state.json` and its recorded checkpoint hash before continuing.

## Verification and current handoff

At installation, the existing task had saved through 130,000 updates and was
still running; the continuous supervisor was observed running and waiting. A
checkpoint from the active run passed read-only format, dataset, runtime, code
and CUDA-state preflight. A Windows smoke test verified the persistent STOP
marker prevents a training launch. A separate dummy virtual-environment
process-tree test verified that `taskkill /T` stops both launcher and child.
Unit tests cover checkpoint/config validation, report-hash use, stop before
launch, recording a completed segment and stopping an owned process tree.
The predecessor completed at 150,000 updates. The supervisor verified the
checkpoint against its completed report (SHA-256
`03d32b8c99a1fe4a6e20af39e71d40ed0c30acd4bd0990fcdecefd1f0b7d6fb0`) and
started the 150,000-to-160,000 segment. GPU activity was observed after the
handoff. Completion of the 160,000 checkpoint and later segments has not yet
been observed; the sign-in recovery trigger has not been exercised.
