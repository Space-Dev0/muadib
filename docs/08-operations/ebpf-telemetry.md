# Optional Linux TCP telemetry

`ops.network_probe` provides the portable application-level TCP echo measurements.
`ops.ebpf_tcp` optionally complements those observations on a Linux endpoint:

```sh
python -m ops.ebpf_tcp --port 22 --seconds 10
```

Use the actual SSH transport port, or the application TCP port when measuring a
direct connection. Run this command concurrently with the workload in another
terminal; it does not start the workload. In Python,
`collect_retransmissions(port=22, seconds=10)` returns the same JSON-compatible
dictionary. This call blocks for the collection period. The application probe
remains usable when this optional collector fails or is absent.

The collector needs Linux, a compatible installed `bpftrace`, the kernel TCP
retransmission tracepoint, and permissions to load/attach BPF programs. The
required capabilities and kernel configuration vary; privileged execution may
be necessary. It does not install packages, change kernel settings, or escalate
privileges. On Windows, collect application-level measurements or run the probe
on a suitable Linux endpoint. A WSL instance only observes traffic visible in
its own kernel/network namespace; it is not automatically a view of Windows.

The counter is scoped to **all host TCP sockets** whose local or remote port
matches the requested port. It does not identify a process, flow, user, job, or
peer. Other SSH sessions on the same port contribute, and multiple forwarded
jobs may share a single SSH transport. An SSH-forwarded loopback port describes
a different socket from the outer SSH port. Tailscale may carry encrypted
traffic over UDP, so this TCP hook is not a measurement of VPN packet loss.
No payload or address is captured by the collector.

`tcp_retransmit_events` counts tracepoint events, not lost packets, retransmit
bytes, throughput, or one-way latency. `dropped_events` and the exact kernel
attachment interval are unknown; the reported timestamps bound the userspace
collector process and include setup/teardown. Collector wall time is reported,
but its CPU overhead is not measured. Compare with a separately measured
no-collector baseline before making performance claims. Warnings, missing
privileges, unsupported systems, timeout, and malformed output produce
`available: false` and a null count, never an invented zero. Even a successful
count cannot prove that no events were missed. Record the selected endpoint,
port, transport, and concurrent workload alongside the output.
