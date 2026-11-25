# One-shot SSH inference launcher

`ops.inference_launcher` starts local SSH forwards to inference workers,
verifies that each endpoint responds to a harmless invalid request, executes a
fixed plan with the adaptive coordinator, then terminates its own SSH
processes. By default it connects to already-running workers. An opt-in
`start_worker` configuration starts a launcher-owned worker as described below.
It does not install dependencies or schedule training.

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
hostnames, or `user@hostname`; usernames may contain spaces. Other special
characters require an SSH alias.
Without an explicit `control_path`, the launcher disables implicit SSH
multiplexing so its child owns the forward it terminates. `--startup-timeout`
applies separately to each owned-worker startup and each forward readiness
exchange, not to the entire multi-worker startup.

The readiness check confirms an endpoint speaks the expected error protocol; it
does not authenticate the worker application or prove GPU health. Verify SSH
destination and local port ownership. Incomplete coordinator runs remain on
disk and return exit code 2. Timeouts can leave remote GPU work running; follow
the uncertain-retry procedure in the coordinator documentation before resuming.
Startup failures return 1. Only tunnels started by this launcher are terminated
on normal return or Python exceptions; an uncatchable process kill can leave
SSH children behind, so inspect them before relaunching. Optional network probe
files are not launched or refreshed by this launcher.

## Optional SSH-owned remote workers

Add a `start_worker` object to an individual tunnel to launch its worker over
SSH. Install the project and a compatible Python/PyTorch environment and copy
the trusted checkpoint beforehand. `python` is the remote Python executable;
`cwd` is the installed project directory containing `ops/`, and `checkpoint`
is its checkpoint path. For example (paths are illustrative):

```json
{"worker_id":"node1", "host":"user@laptop1", "ssh_port":22, "remote_port":53211,
 "start_worker":{"platform":"linux", "python":"/home/user/muadib/.venv/bin/python",
                 "cwd":"/home/user/muadib", "checkpoint":"/home/user/model.pt",
                 "format":"warmstart", "device":"cuda", "lease_seconds":3600}}
```

Use `"platform":"windows"` and Windows paths for a Windows OpenSSH server.
Windows Python executable paths may contain spaces; Linux executable paths must
not. Other paths are encoded, and Python executable shell metacharacters are
rejected. The remote
Python must support `hashlib.file_digest` (Python 3.11 or newer). Worker source
and checkpoint must already be available on that host.

The remote Python process checks the checkpoint SHA-256 against the worker
configuration and announces a random ownership marker only after binding the
requested loopback listener. Its watchdog exits that same process on SSH stdin
EOF, and its lease timer exits it even if a disconnected SSH channel does not
deliver EOF. On Windows the watchdog polls the SSH pipe without a blocking
stdin read, avoiding a NumPy native-import deadlock. The launcher closes the
channel on cleanup and waits briefly for
an exit marker; if cleanup cannot be confirmed it warns rather than killing a
PID or a process using the same port. A pre-existing listener is not killed.
Set `lease_seconds` (30–86,400) longer than startup plus the entire expected
run; expiry during inference can leave the current job uncertain. SSH host-key
verification and authenticated access are still required. The marker and
checksum validate the intended setup over a trusted SSH connection; they are
not cryptographic attestation against a malicious remote host.
The worker protocol is unauthenticated even on loopback: other users or
processes on a shared worker host may connect to its listening port. The
ownership token is included in the remote launch command and must not be
treated as a secret or as authentication against local users. Run workers only
on hosts whose other users and processes are trusted.

Process termination, network failure or coordinator crash may prevent a cleanup
acknowledgment. Inspect the remote host before rerunning an uncertain job; the
lease bounds the intended worker lifetime but is not proof of cleanup on an
unresponsive or suspended machine. The Windows command syntax targets OpenSSH
with `cmd.exe` as the remote shell; PowerShell is not verified. Windows and
Linux require separate live validation with the exact remote shell and Python
environment in use.

A tunnel entry may optionally include `"control_path": "/absolute/path/to/socket"`
to use an already authenticated OpenSSH ControlMaster via `ssh -S -O forward`. It must be
an existing non-symlink Unix socket. The socket owner controls that SSH session;
only provide one you trust and whose destination matches the configured host.
This does not alter authentication or host-key policy. The launcher does not
own or close the ControlMaster itself.
The launcher requests `ssh -O cancel -L ...` for its configured multiplexed
forward during cleanup while leaving the master running. It warns if cancellation fails. Avoid concurrently allocating the same
local port from other processes: a preflight check cannot eliminate that race.

If a ControlMaster forward request times out, its ownership is uncertain: the
master may have installed the forward after the client stopped waiting, or a
concurrent client may own that port. The launcher does not cancel an
unconfirmed forward. Inspect the master and listening port and coordinate with
its owner before making changes; only cancel or stop the master when its owner
has established that doing so will not affect other sessions.

Owned-worker startup errors report only a few fixed diagnostic categories, such
as checkpoint mismatch or SSH authentication failure. Other remote output is
not echoed because it may contain paths, commands or ownership markers.

To report a finished or incomplete session and independently check each planned
result's PNG and digest against the plan and recorded stratum counts, run:

```sh
uv run python -m ops.benchmark_report --verify-results outputs/network-run
```

Omitting `--verify-results` reports recorded counts without independently
checking result files. Both modes validate that recorded stratum counts match
the plan and overall counts. Verification checks consistency and file integrity;
it does not assess image semantics or attest the remote worker's identity.
