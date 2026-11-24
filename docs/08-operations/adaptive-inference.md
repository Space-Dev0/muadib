# Resumable inference coordinator

`ops.adaptive_inference` adds a frozen job plan and durable attempt log to the
existing independent-image worker protocol. Each worker serves one inference at
a time and has one loaded checkpoint. Jobs may target different checkpoints,
but a worker is eligible only when its checkpoint SHA-256 matches the job.
This is independent image-level inference, not distributed model training or a
claim of statistical equivalence across different hardware.

Create a plan JSON, replacing the digest with the checkpoint file's lowercase
SHA-256. The current worker protocol supports only red and blue color requests.
IDs and strata contain letters, digits, underscores and hyphens only.

```json
{"schema_version": 1, "jobs": [
  {"id": "red-1", "stratum": "balanced", "checkpoint_sha256": "REPLACE_WITH_64_HEX_CHARACTERS", "color": "red", "seed": 1},
  {"id": "blue-2", "stratum": "balanced", "checkpoint_sha256": "REPLACE_WITH_64_HEX_CHARACTERS", "color": "blue", "seed": 2}
]}
```

Start compatible workers and forward their ports to the coordinator using SSH.
Configure the local tunnel ports, not public network addresses:

```json
{"workers": [
  {"id": "node1", "port": 53211, "checkpoint_sha256": "REPLACE_WITH_64_HEX_CHARACTERS", "estimate_seconds": 60},
  {"id": "node2", "port": 53212, "checkpoint_sha256": "REPLACE_WITH_64_HEX_CHARACTERS", "estimate_seconds": 60}
]}
```

```sh
uv run python -m ops.adaptive_inference --plan plan.json --workers workers.json \
  --output outputs/network-run --policy adaptive
uv run python -m ops.adaptive_inference --plan plan.json --workers workers.json \
  --output outputs/network-run --policy adaptive --resume
```

`round-robin` is a baseline. `adaptive` selects an available compatible worker
using a configured prior or the exponentially smoothed wall time of prior
successful requests. Optionally add both `probe_file` and `probe_port` to a
worker to consume a recent `ops.network_probe` output for the corresponding
SSH-forwarded loopback echo endpoint. The routing score adds median echo RTT to
the wall-time estimate as a heuristic, even though request wall time may already
include network overhead. It is not a calibrated predictor and echo RTT is not
one-way link latency. Missing, partially failed, wrong-endpoint or stale telemetry is
ignored; it never excludes a worker. A probe must use schema 1, have valid
configuration and at least five successful, consistent recorded attempts; its
reported median must agree with its recorded measurements. This minimum is a
quality filter, not evidence of a stable latency estimate. The scheduler does
not launch probes.

Every attempted job is recorded in `events.jsonl`, including failures and
retries; unfinished jobs remain in the fixed plan. Each completed job has a
verified PNG and a JSON record under `results/<id>/`. `summary.json` reports
planned, completed and missing counts per stratum, and wall time for the most
recent session only; it excludes startup, resume validation, and summary writing. A resumed session verifies the plan, PNG and digest and
skips completed jobs. It tries each compatible worker at most once per job per
session. An incomplete run exits 2 and may be resumed; errors exit 1.

Every new `run_started` event stores a versioned session configuration: policy,
request timeout, telemetry freshness, worker IDs, local ports, expected
checkpoint hashes, priors and whether probes are configured. It also records
the coordinator source SHA-256 when available. SSH hosts, credentials and probe
file paths are omitted. Changing the workers or policy on resume is allowed for
recovery, but mixed-policy sessions are not directly comparable as experiments.
The summary separates setup time and result-validation time from dispatch-segment
time; elapsed-before-summary excludes writing the summary and final event.

After an abrupt process stop, check that no coordinator is running before
removing a stale `.coordinator.lock`. An unterminated final event is archived
byte-for-byte as `events.torn-<id>.jsonl` before the working journal is repaired;
missing jobs are treated as uncertain and require `--allow-uncertain-retry`
after inspecting the archive and workers. Complete malformed or middle lines
stop recovery and remain unchanged. A published result can precede the logged
success event if the process stops at that exact point; resume preserves the
result and records the dangling start as abandoned rather than inventing an
outcome. Use SSH authentication and forwarding: checksums validate file
integrity, not peer identity.

## Timeouts, retries and trust

A client timeout, broken connection, malformed response or other ambiguous
failure does **not** stop an already submitted GPU job. The coordinator marks
that attempt uncertain, quarantines its worker for the rest of the session and
does not automatically retry that logical job. A definite connection refusal
can be retried on another worker. Check the remote worker before resuming an
uncertain job; after accepting the risk of duplicate compute, add
`--allow-uncertain-retry` to `--resume`. An abrupt coordinator stop also makes
its unfinished attempts uncertain. This acknowledgment cannot cancel a remote
job or guarantee that only one copy runs. Other coordinator processes can also
submit duplicate work; the lock applies only to this run directory.

If receiving a response succeeds but local image verification or publication
fails, the attempt is recorded as a publication failure rather than an
uncertain remote execution. It is not retried automatically within the same
session; resolve the local issue and resume the run.

The worker reports device, software versions and source digests in its runtime
metadata, which is saved with each result. These are worker claims, not
cryptographic attestation. A loopback port can be occupied by another local
process; SSH protects a correctly established tunnel but the protocol has no
per-request authentication or independent remote-identity verification. Verify
port ownership and the SSH destination before benchmarking. The request timeout uses one monotonic deadline for connect, send, and all
response reads. It does not limit remote execution or guarantee cancellation.
