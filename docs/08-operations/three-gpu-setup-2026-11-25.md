# Three-GPU setup and worker lifecycle, 25 November 2026

The coordinator is CPU-only and does not run GPU jobs. Three remote GPUs have
been verified: two Windows RTX 3050 laptops and NewAurora's Linux RTX 2050.
Each has CUDA-enabled PyTorch and a copy of the same 480,000-update warmstart
checkpoint with SHA-256
`f29d75e9c85ffed5c0f3e184d37b0069586e6d75184f253bea53ba46e56cf459`.
NewAurora's environment uses PyTorch 2.11.0+cu128; both Windows workers use
2.9.1+cu128. Cross-version bitwise equivalence is not assumed. NewAurora
now accepts the coordinator's public SSH key; its host key was independently
confirmed and is checked strictly.

The WSL endpoint has not been verified as a GPU worker.

An optional launcher-managed worker was tested end to end on NewAurora and across
all three verified GPUs (`gpu-1` and `gpu-2` Windows RTX 3050 laptops, plus
`gpu-3` NewAurora Linux RTX 2050):

- **Three-GPU smoke check (`three-gpu-owned-20261125`):** 3/3 planned jobs
  completed (`round-robin`, `26.40s` segment wall time, 0 missing).
- **Three-GPU 18-job matched runs (`three-gpu-18-round-robin-20261125` and
  `three-gpu-18-adaptive-20261125`):** both completed 18/18 jobs with 0 missing
  and passed `ops.benchmark_report --verify-results` artifact validation.
  - `round-robin`: `25.19s` segment wall time (`0.714` images/s), median
    request duration `1.361s`, p95 `11.822s`, worker split `1 / 3 / 14`
    (`gpu-1 / gpu-2 / gpu-3`).
  - `adaptive`: `25.73s` segment wall time (`0.700` images/s), median
    request duration `1.436s`, p95 `11.794s`, worker split `1 / 3 / 14`
    (`gpu-1 / gpu-2 / gpu-3`).
- **GPU-3-only 18-job baseline (`single-gpu3-18-20261125`):** `gpu-3` alone
  completed 18/18 jobs in `26.57s` (`0.677` images/s, median `1.418s`, p95
  `1.574s`).
- **Three-GPU vs GPU-3-only 36-job comparison:** all three runs completed
  36/36 jobs with 0 missing and passed `ops.benchmark_report --verify-results`:
  - `single-gpu3-36-20261125` (`gpu-3` alone): `51.73s` (`0.696` images/s,
    median `1.401s`, p95 `1.465s`).
  - `three-gpu-36-round-robin-20261125` (3 GPUs, `round-robin`): `46.50s`
    (`0.774` images/s, median `1.405s`, p95 `12.484s`, split `2 / 6 / 28`).
  - `three-gpu-36-adaptive-20261125` (3 GPUs, `adaptive`): `46.63s`
    (`0.772` images/s, median `1.448s`, p95 `12.263s`, split `2 / 6 / 28`).
- **Controlled mid-run disconnect & resume (`three-gpu-fault-resume-20261125`):**
  The run was described as using a `30s` lease on `gpu-2`; the saved
  ledger records EOF on `blue-9705` but does not record the lease setting or
  establish the cause. The coordinator recorded 1 uncertain attempt, quarantined `gpu-2`, completed the other 35 jobs on `gpu-1` and
  `gpu-3` in `47.38s` (exit code 2), and then `--resume --allow-uncertain-retry`
  reused the 35 verified PNGs and completed `blue-9705` (36/36, 0 missing,
  exit code 0, verified with `ops.benchmark_report --verify-results`).

Because each policy dispatches to whichever worker is idle and `gpu-1` took ~25s
on its first image while `gpu-3` averaged ~1.3–1.4s per image, both policies
produced the same `1 / 3 / 14` (18-job) and `2 / 6 / 28` (36-job) worker splits.
In the single 36-job comparisons, the recorded three-worker segment was ~10%
shorter than `gpu-3` alone (`46.50s` / `46.63s` vs `51.73s`).
These are single-run integration and throughput checks without live
`network_probe` RTTs, not a model-quality evaluation or statistical evidence of
an adaptive scheduling speedup over `round-robin`.

A separate live eBPF recheck (`ops.ebpf_tcp`, port 22, requested interval
3 seconds) reported `available: true`, zero retransmission events and 3.531
seconds collector elapsed. The [raw capture](network-report/ebpf-live-recheck.json)
records that dropped-event counts and the exact attachment window are unknown;
its host TCP port scope does not attribute events to a job, process or VPN peer.
The earlier reported 3.46-second observation has no retained raw capture. A
zero TCP count does not establish zero Tailscale UDP packet loss.

## Independent recheck

A fresh three-worker run and a separate test of the reusable local launcher
command each completed 3/3 planned images. After each run, checks found no
listener on any of the three test worker ports. Those are integration checks,
not statistical speedup or image-quality evaluations. The local-only launcher
configuration, which contains private addresses but no passwords, is under
`outputs/network-project-one-shot-20261124/three-gpu-config/`.

The networking report includes a sanitized machine-readable
[benchmark summary](network-report/benchmark-summary.json). Full plans,
attempt logs and generated images remain in the ignored local `outputs/`
directory; the summary is not a substitute for distributing those raw files.
