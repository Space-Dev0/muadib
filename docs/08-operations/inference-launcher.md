# One-shot SSH inference launcher

`ops.inference_launcher` starts local SSH forwards to **already running**
`ops.distributed_inference worker` processes, verifies that each endpoint
responds to a harmless invalid request, executes a fixed plan with the
adaptive coordinator, then terminates its own SSH processes. It does not
install, start, stop or restart remote workers and does not schedule training.

Configure the plan and workers as described in [adaptive inference](adaptive-inference.md).
Use unique local ports in the worker list. Create `tunnels.json`, one entry for
each worker ID:

```json
{"tunnels": [
  {"worker_id": "node1", "host": "user@laptop1", "ssh_port": 22, "remote_port": 53211},
  {"worker_id": "node2", "host": "user@laptop2", "ssh_port": 22, "remote_port": 53211}
]}
```

```sh
uv run python -m ops.inference_launcher --plan plan.json --workers workers.json \
  --tunnels tunnels.json --output outputs/network-run --policy adaptive
# Once workers are back, resume the same frozen plan:
uv run python -m ops.inference_launcher --plan plan.json --workers workers.json \
  --tunnels tunnels.json --output outputs/network-run --policy adaptive --resume
```

SSH uses local credentials and host-key verification, with `BatchMode=yes` and
`ExitOnForwardFailure=yes`. Set up SSH keys and trust host keys ahead of time;
no password or host-key bypass is supported. Host values are SSH aliases,
hostnames, or `user@hostname` without whitespace. For usernames containing
spaces or other special characters, configure an SSH alias locally.
Without an explicit `control_path`, the launcher disables implicit SSH
multiplexing so its child owns the forward it terminates. `--startup-timeout`
applies to each worker's forward setup and readiness exchange, not to the
entire multi-worker startup.

The readiness check confirms an endpoint speaks the expected error protocol; it
does not authenticate the worker application or prove GPU health. Verify SSH
destination and local port ownership. Incomplete coordinator runs remain on
disk and return exit code 2. Timeouts can leave remote GPU work running; follow
the uncertain-retry procedure in the coordinator documentation before resuming.
Startup failures return 1. Only tunnels started by this launcher are terminated
on normal return or Python exceptions; an uncatchable process kill can leave
SSH children behind, so inspect them before relaunching. Optional network probe
files are not launched or refreshed by this launcher.

A tunnel entry may optionally include `"control_path": "/absolute/path/to/socket"`
to use an already authenticated OpenSSH ControlMaster via `ssh -S -O forward`. It must be
an existing non-symlink Unix socket. The socket owner controls that SSH session;
only provide one you trust and whose destination matches the configured host.
This does not alter authentication or host-key policy. The launcher does not
own or close the ControlMaster itself.
The launcher requests `ssh -O cancel -L ...` for its configured multiplexed
forward during cleanup while leaving the master running. It warns if cancellation fails. Avoid concurrently allocating the same
local port from other processes: a preflight check cannot eliminate that race.

To report a finished or incomplete session and independently check each planned
result's PNG and digest against the plan and recorded stratum counts, run:

```sh
uv run python -m ops.benchmark_report --verify-results outputs/network-run
```

Omitting `--verify-results` reports recorded counts without independently
checking result files. Both modes validate that recorded stratum counts match
the plan and overall counts. Verification checks consistency and file integrity;
it does not assess image semantics or attest the remote worker's identity.
