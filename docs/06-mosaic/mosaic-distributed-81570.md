# Distributed 500,000-update continuation

The user requested at least 500,000 updates per matched condition across two
laptops, with automatic resumption. The assignment is low-balanced and
low-imbalanced on the second laptop, and high-balanced and high-imbalanced on
the original laptop. Each laptop trains its assigned cells sequentially.
The first laptop's original four-cell task was allowed to keep running, rather
than interrupted: both low cells had completed 150k and the high cells were
still in progress when the split was prepared. Its high-cell supervisor waits
for the old task before using that GPU. The two completed low 150k checkpoints
and reports were transferred to the second laptop and their hashes matched:

- low-balanced: `1ce1f7e64d58044b637647acc2207499c0f828cb905f4350b2a0aafc2649cd64`
- low-imbalanced: `e9ec2ff46412adf0b2e0a8b50dbf4553e5cfc9d32a7aff6605328ec93e2456b1`

The dataset manifest hash is
`48bf70c4d43c07bdc563cba1f585ea618e862047635257a4ce63f67c9bd45d37`.
The second laptop has a different driver; moving saved state and RNG streams
between machines does not guarantee bitwise continuation. The experiment is
still one paired training-seed block, not a replicated MOSAIC result.

`exposure.mosaic_distributed_supervisor` runs bounded segments up to 10,000
updates with separate output directories. It checks the source checkpoint and
report hashes, cell, steps, architecture, CUDA device count, PyTorch/Python
versions and code fingerprints before continuing. Successful segment reports
and their checkpoint hashes identify the newest completed segment. After an
interruption, an incomplete directory is preserved; a loadable checkpoint with
no final report can be recovered after schema and configuration checks, but
its calculated hash has no independent report hash. The log identifies this
weaker provenance. Partial or invalid files are skipped. If an initial cell
never started, the supervisor starts a new matching initial run. A file lock,
Windows task `IgnoreNew` setting, and an orphan-process check prevent overlap.
These mechanisms are tested but cannot guarantee recovery from hardware, disk,
Windows policy, or incompatible-software failures.

The scheduled task is triggered at sign-in and every five minutes for one
year, with multiple instances ignored. Ordinary sleep suspends a running
process, which should continue when Windows wakes. After reboot, the task
requires the configured Windows user to sign in; it does not wake the laptop,
log in automatically, or override shutdown and sleep policies. The supervisor
restarts from the most recent verified saved point, so interrupted updates
since that checkpoint may be repeated. A `STOP` file inside the workspace's
`outputs/mosaic-distributed-81570` directory stops the supervisor and its
owned segment; the existing continuous-training STOP marker is separate.

Original laptop monitoring:

```powershell
Get-ScheduledTask MuadibMosaicMatched81570,MuadibMosaicDistributed81570-high | Select-Object TaskName,State
Get-Content "$env:USERPROFILE\muadib\mosaic-distributed-81570-high.log" -Tail 30
```

Second laptop monitoring after setup:

```powershell
Get-ScheduledTask MuadibMosaicDistributed81570-low | Select-Object TaskName,State
Get-Content "$env:USERPROFILE\muadib\mosaic-distributed-81570-low.log" -Tail 30
```

The original laptop's high supervisor was installed and observed waiting while
the original task continued. On the second laptop, uv's Python executable was
blocked by Windows application control, so the official signed Python 3.13.6
package was installed using winget. A CUDA wheel download failed after
DNS/timeouts and an SSH disconnect. After reconnection a separate setup task,
`MuadibMosaicSecondSetup81570`, was installed and observed running. It retries
dependencies at sign-in and every 30 minutes for seven days, verifies both
transferred checkpoints, performs and verifies a one-step CUDA continuation,
and installs the low-cell supervisor only if those steps succeed. The low-cell
task has not yet been observed running; installation of the setup task alone is
not evidence that low-cell training has started.

During setup, stopping the first scheduled supervisor left detached Python
processes, and subsequent starts encountered its held lock. The scheduled task
was disabled, only processes whose command line named the distributed
supervisor were stopped, and the task was re-enabled. The original model
trainer was not stopped. The task and one Python launcher/child process tree
were then observed running together with the original training task. The
second laptop's setup task was observed running with an active connection to
the PyTorch download host; its training task was not yet present.

On a later check, the original laptop had completed high-balanced at 500,000
updates and was continuing high-imbalanced beyond 320,000. The second laptop
had installed CUDA PyTorch, but NumPy timed out. Setup was restarted with a
180-second HTTP timeout and the first laptop's exact NumPy, Pillow and SciPy
versions. The low-cell task must still pass its checkpoint migration smoke
test before it starts.

On the next check, the second laptop's network setup had failed again. The
three pinned Windows wheels were downloaded from PyPI on the local machine,
transferred, and checked by SHA-256. Setup was updated to use these local
wheels when present. It installed all three, verified CUDA and both 150k
checkpoint hashes, and completed a one-update migration smoke test.
`MuadibMosaicDistributed81570-low` was then observed running a low-balanced
150,000-to-160,000 segment with GPU activity. Cross-machine continuation is
not guaranteed bitwise identical despite restored RNG state.

On 2026-11-17 the user requested moving the unfinished low-cell work back to
the first laptop because the second laptop was unreachable. The first laptop
had completed both high cells at 500,000 updates and retained the original
verified 150,000-update low-cell checkpoints. Its new
`MuadibMosaicDistributed81570-low` task was observed running the
low-balanced 150,000-to-160,000 segment on CUDA. The target remains 500,000
updates per cell and checkpoints remain spaced at 10,000 updates.
The completed high-cell supervisor was disabled to avoid competing periodic
checks; its checkpoints remain intact.

The second laptop's later progress and task state could not be inspected, and
its automatic low-cell task could not be disabled while it was offline. If it
comes back online, disable or stop that task before it resumes the same work;
preserve its checkpoints and verify their provenance before comparing or using
them. The first laptop may repeat updates already completed on the second.
