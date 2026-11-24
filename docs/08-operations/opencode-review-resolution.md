# Networking review follow-up

The 25 November review examined the coordinator, echo probe and benchmark
reporter. Its findings were checked against the code and are addressed as
follows. Existing pilot records have not been altered.

- **Recovery:** local publication failures are distinguished from uncertain
  remote failures. A torn final journal record is archived before recovery;
  incomplete jobs require an explicit acknowledgment before uncertain retries.
  Malformed complete records still fail closed.
- **Telemetry:** probe files must be regular, bounded, fresh and consistent,
  with at least five successful samples. Echo payloads are random and the
  server timeout, pacing and percentile method are identified explicitly.
- **Reproducibility:** each new session records the coordinator version,
  policy, worker configuration and timeout settings. Resuming can change
  workers or policy; the reporter identifies such changes. Setup and result
  validation are timed separately from dispatch.
- **Reporting:** stratum totals are checked against the plan. Use
  `uv run python -m ops.benchmark_report --verify-results RUN_DIRECTORY` to
  check each recorded image, checksum and completion count. Request timing,
  percentile conventions and legacy missing metadata are identified.
- **Launcher:** SSH startup has an absolute deadline; standalone tunnels do
  not silently attach to an existing ControlMaster. Explicitly shared masters
  use separately registered and cancelled forwards.

The adaptive score remains an exploratory heuristic: it can add echo RTT to an
already network-inclusive request duration, uses per-worker rather than
per-stratum history, and resets history on resume. It has not been shown to
improve performance. The reporter requires a completed session summary; it
does not yet provide an events-only report for a crashed session. A second
interrupt during thread shutdown and a stalled filesystem `fsync` remain
potential operational hazards. These limits do not demonstrate any Stage 0
research result or a statistically established network speedup.
