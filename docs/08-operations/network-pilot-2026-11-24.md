# Multi-device network pilot — 24 November 2026

This is a short connectivity smoke test, not an evaluation of the proposed
adaptive scheduler. The coordinator sent 20 sequential 1,024-byte TCP echo
requests per laptop through an authenticated SSH local-forwarding tunnel over
Tailscale. The configured gap was 150 ms and the socket-operation timeout was
3 s. The probe has since been fixed to enforce an overall attempt deadline;
these earlier records used the prior per-operation timeout behavior.
Times are application echo round-trip times and include endpoint and tunnel
processing. Median and p95 are nearest-rank estimates; RTT variation is the
absolute difference between successive successful RTTs. Five hosts participated,
but one was a WSL endpoint and was not verified as a GPU worker.

| Endpoint | Successful probes | Median RTT (ms) | p95 RTT (ms) | Median RTT variation (ms) |
|---|---:|---:|---:|---:|
| Windows GPU A | 20/20 | 90.40 | 238.58 | 24.68 |
| Windows GPU B | 20/20 | 289.79 | 391.92 | 70.09 |
| Fedora | 20/20 | 70.28 | 86.58 | 1.72 |
| NewAurora | 20/20 | 105.85 | 279.23 | 37.93 |
| WSL endpoint | 20/20 | 89.55 | 107.70 | 6.87 |

No probe timed out. With only 20 observations per endpoint, percentiles are
exploratory; differing run windows and background traffic make this neither a
fair scheduler comparison nor evidence of causal improvements. These samples
are not packet-loss measurements. Goodput from this paced tiny-payload test is
not a link-capacity estimate. The raw local records are under
`outputs/network-project-one-shot-20261124/`, which is not required to be
tracked in Git.

Windows GPU A reported an RTX 3050 and CUDA-capable PyTorch. Windows GPU B
reported an RTX 3050 but its PyTorch CUDA state was not verified in this
inventory. Fedora has an RTX 3050 visible on PCI, but `nvidia-smi` and PyTorch
were unavailable during this check. NewAurora's isolated CUDA worker was later
verified on its RTX 2050. GPU visibility alone does not
establish a usable training or inference worker. The WSL endpoint's GPU status
was not established.

The new [resumable coordinator](adaptive-inference.md) and
[optional eBPF collector](ebpf-telemetry.md) have local tests. The eBPF collector
has not been demonstrated against a live kernel because `bpftrace` was not
available locally. A remote GPU inference comparison requires matching model
checkpoint hashes, compatible worker runtimes, a frozen workload, and repeated
baseline/adaptive runs. See the [measurement plan](network-metrics-plan.md)
for the complete definitions and controls. This connectivity pilot does not
change the status of the diffusion or Stage 0 research gates.

## GPU inference integration smoke test

The coordinator completed two planned requests, red and blue, through an SSH
tunnel to Windows GPU A using its existing 480,000-update warmstart checkpoint.
Both returned images passed PNG, size and digest validation; no request failed
or remained unfinished. The segment took 49.03 seconds (0.041 completed images
per second, including dispatch and response handling). This is a two-image,
one-worker integration check, not a comparison of scheduler policies, a model
quality evaluation, or a claim of speedup. The local plan, per-attempt ledger,
image records and summary are in `outputs/network-project-one-shot-20261124/gpu-smoke/`.

## Two-GPU matched-workload smoke comparison

The same SHA-256-verified 480,000-update checkpoint was loaded on Windows GPU A
(RTX 3050) and NewAurora (RTX 2050). Each worker first completed one warm-up
image. Then each policy processed the same eight jobs (four noise seeds, two
colors) once, in the fixed order round-robin followed by adaptive:

| Policy | Completed | Dispatch segment (s) | Images/s | Jobs on GPU A / NewAurora |
|---|---:|---:|---:|---:|
| Round-robin | 8/8 | 21.719 | 0.368 | 1 / 7 |
| Adaptive | 8/8 | 21.510 | 0.372 | 1 / 7 |

There were no failed or unfinished attempts in either run. Both policies
assigned every corresponding job to the same worker and produced identical
PNG hashes for the same job in this particular comparison. This observation
does not guarantee cross-device reproducibility. Both schedulers dispatch to
available workers, explaining why the faster-responding endpoint received
more jobs in these short runs.

These numbers are an integration smoke test, **not evidence that adaptive
scheduling improves performance**: there is only one run per policy, the order
was fixed, the workload was small, and echo-probe input was not configured for
the adaptive run. The Windows worker used PyTorch 2.9.1 and Python 3.13; the
Linux worker used PyTorch 2.11.0 and Python 3.12. Their worker source file
hashes also differ, although the three recorded sampler/model module hashes
match. Software, hardware, path and system load were not controlled. Segment
timing excludes worker startup, warm-up and coordinator setup, so it is not
end-to-end deployment latency. The raw plans, events, results and runtime
metadata are retained under `outputs/network-project-one-shot-20261124/`.

The later [one-shot launcher](inference-launcher.md) was also tested with two
GPU workers and two image jobs. Both jobs completed, and both local forwarding
ports were confirmed closed afterward, including the forward controlled by an
existing SSH ControlMaster. The test workers were stopped after the run. The
launcher starts tunnels to existing workers; it does not start remote workers.
