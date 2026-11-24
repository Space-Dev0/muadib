# Networking prototypes

These are experimental protocols, not a completed Stage 0 result. They do not encrypt or authenticate peers: use only over an authenticated tunnel or isolated trusted network. SHA-256 detects corruption, not malicious modification by an attacker. Do not expose prototype ports publicly.

The newer [resumable, checkpoint-aware coordinator](../docs/08-operations/adaptive-inference.md)
adds a frozen workload, attempt ledger, conservative handling of uncertain
failures, and optional echo-based routing. The [one-shot launcher](../docs/08-operations/inference-launcher.md)
sets up SSH forwards to already-running workers. The [network pilot](../docs/08-operations/network-pilot-2026-11-24.md)
reports the first short multi-host connectivity measurements; the optional
[eBPF collector](../docs/08-operations/ebpf-telemetry.md) requires a supported
Linux environment and suitable privileges.

## Image-level inference across hosts

`ops.distributed_inference` distributes independent `(color, seed)` jobs for
the MOSAIC PilotUNet probe or warmstart checkpoint formats. It is not
data-parallel training, and does not split one image's diffusion steps. Install
the same project revision and compatible PyTorch/CUDA runtime on both workers;
copy the same trusted checkpoint to each and verify its SHA-256. On each
machine run (choose `probe` instead if appropriate):

```sh
uv run --extra train python -m ops.distributed_inference worker \
  --checkpoint PATH_TO_CHECKPOINT --format warmstart --port 53211 --device cuda
```

From the coordinator, establish a separate SSH local forward to each worker
and keep those sessions alive:

```sh
ssh -N -L 53211:127.0.0.1:53211 USER@HOST_A
ssh -N -L 53212:127.0.0.1:53211 USER@HOST_B
```

Then run a small fixed-seed test using the SHA-256 printed by the worker and a
new output directory:

```sh
uv run python -m ops.distributed_inference run \
  --worker-port 53211 --worker-port 53212 --checkpoint-sha256 SHA256 \
  --seed 1 --seed 2 --output outputs/distributed-inference-example
```

The coordinator retries another endpoint after an error, verifies payload
digest and PNG dimensions, refuses duplicate jobs and pre-existing output
directories for new runs, and saves an immutable job plan. Each completed image
and its metadata are published together in a job directory. After a disconnect,
repeat the same command with `--resume`: the coordinator checks the plan, PNGs
and hashes, reuses valid completed jobs and runs only missing jobs. Different
seeds, colors, job order or checkpoint hashes are rejected on resume. The
aggregate manifest records timing, worker assignments, image hashes and worker
runtime/source fingerprints when every job completes. `segment_wall_seconds`
is only the most recent invocation, not the total across interrupted attempts.

A coordinator lock prevents concurrent resumes. A hard process termination may
leave `.coordinator.lock` or `.staging-*` directories behind. After verifying no
coordinator is still running, remove the stale lock before resuming; staging
directories are ignored and can be inspected and removed manually. Atomic
renames protect against partial published job directories during normal process
interruption; power-loss durability is not guaranteed. A lost reply can cause a
job to be computed twice; no exactly-once execution is claimed. CUDA outputs
need not match bitwise across devices. Total wall time and per-job round trips
are not GPU idle measurements, and no speedup has been established. Tests
cover local sockets and simulated failover, not two physical GPU hosts.

Workers listen only on loopback. TCP and SHA-256 do not authenticate peers;
use SSH forwards on trusted machines. A matching checkpoint hash is not proof
that a checkpoint is safe or runtime dependencies are identical. Load only
trusted checkpoints and compare recorded fingerprints before interpreting
cross-host comparisons.

## Unconditional scene streaming

`scene_stream` sends batches of CPU-rendered 64×64 PNG images and JSON annotations. It is separate from the training experiment; the demonstrated distribution is unconditional.

```sh
uv run python -m ops.scene_stream serve --host 127.0.0.1 --port 53121 --count 300 --batch-size 8
uv run python -m ops.scene_stream receive --host 127.0.0.1 --port 53121 --batch-size 8 --metrics outputs/stream-metrics.jsonl
```

Start the producer in one terminal, then the receiver in another. On the receiver, `--batch-size` labels the report and must match the producer. `receive_decode_seconds` includes blocking, frame parsing and PNG decoding; it is not GPU idle time or pure link latency. Payload throughput counts PNG bytes only; it excludes JSON and TCP/SSH overhead and is not a measurement of physical link capacity.

`local-benchmark.json` records a short localhost run (300 scenes each for batch sizes 2, 8 and 16). Reproduce with `uv run python -m ops.benchmark --count 300 --output outputs/network-local.json`.

`tunnel-benchmark.jsonl` records one short real inter-machine run in the direction **Linux producer → Windows laptop receiver**, using SSH remote port forwarding over the existing private connection; 300 scenes, seeds 90000–90299, per batch size. Results are 582.0/801.1/925.7 scenes/s and 4.91/6.76/7.81 Mb/s PNG payload for batches 2/8/16 respectively. The receiver's `host` field is loopback on its side of the SSH tunnel, not the remote machine address. These are three individual sub-second trials, not the requested 15-minute LAN benchmark, no error bars or evidence of stable differences. The producer uses the unconditional renderer; no model or GPU participated.

`tunnel-second-laptop-82993.jsonl` records a separate 300-scene-per-setting
trial to the newly added Windows laptop, seed 82993: 555.6/827.4/513.8
scenes/s for batches 2/8/16. These single sequential trials are too short to
rank configurations and do not measure GPU utilization or physical bandwidth.

## Opt-in conditional pilot adapter

`conditional_stream` is request-driven: the trainer chooses each rule and seed, and the producer returns the image, object annotations, rejection attempts and PNG hash. The consumer checks rule-definition order/hash, server Python/Pillow and renderer/rule/measurement source hashes, image checksum, dimensions, and rule satisfaction of returned annotations. Matching these versions is a strict prerequisite; different Python versions are rejected. Checksums and source fingerprints do not authenticate the peer, prove pixel/annotation correspondence against a malicious sender, or verify every dependency.

`pilot_stream.train_via_stream` is an **opt-in wrapper** for `exposure.pilot_train.train`; the existing trainer and its normal version-3 checkpoint format are unchanged. It calls the original trainer and temporarily swaps its process-global `render_conditioned` function. Run it in an isolated process with no other training threads; the wrapper lock only guards other wrapper calls, not unrelated code using the trainer. It restores the function and releases the lock even if a request fails. It makes one synchronous network request per scene, not parallel prefetch or batched trainer requests. It is experimental and does not prove a speedup. It requires a connected socket to a `serve_requests` producer.

Streamed checkpoints use wrapper schema version 4, record the ordered rule hash, producer renderer runtime and adapter/protocol source hashes, and are deliberately rejected by the ordinary version-3 resume path. Resume only through `train_via_stream` with the same provenance. Existing version-3 checkpoints cannot be resumed through the wrapper. Neither versioned provenance nor a check that annotations satisfy a rule establishes equivalence across unverified dependencies or hardware.

The isolated test used a copied source tree on the Windows laptop, Python 3.13.6, PyTorch 2.9.1+cu128, **CPU**, one PyTorch thread, 64×64 images, width 4, four timesteps and fixed seed 19. It ran 2 uninterrupted local updates, 2 streamed updates, 1+1 streamed resume updates, and a failure/lock-release check. Over same-host localhost TCP it compared model/EMA tensors, loss, selected/realized counts, rejection attempts and Python/PyTorch RNG state exactly. All passed; a malformed response restored the original renderer and a subsequent wrapper call succeeded; the normal v3 loader rejected the v4 checkpoint. It did not use the GPU or touch the active training workspace.

```sh
uv run --extra train python -m ops.validate_pilot_stream
uv run --extra train python -m ops.benchmark_pilot_stream --output outputs/cpu-pilot-stream.json
```

`cpu-pilot-stream-benchmark.json` reports a separate one-update CPU run per mode and batch 2/8/16 over same-host localhost TCP. Its round-trip timer covers synchronous remote render, transfer and decoding; it is not GPU idle time. There was one trial per configuration with no warmup, so the timings cannot establish a performance improvement or steady-state throughput. No model training over the inter-machine SSH tunnel has been measured.

## Resumable checkpoint transfer

A longer localhost run, `extended-local-benchmark.json`, processed 300,000
scenes per batch size (900,000 total) in 843.04 seconds, approximately 14.1
minutes of measured transfer time. Batch sizes 2/8/16 gave 1,082.10/1,061.01/
1,059.87 scenes/s and 9.10/8.92/8.91 Mb/s PNG payload. It used the same seeds
per size, includes CPU rendering and encoding, and has one sequential trial per
setting; it does not establish link capacity, GPU idle, training performance,
or a 15-minute cross-laptop result.

`checkpoint_sync` compresses a stable source with gzip, validates compressed and decompressed SHA-256, resumes interrupted compressed transfer from its partial byte length, and publishes only after validation. Existing destination files are never overwritten. It does not perform delta encoding or automatically resume training. The receiver exits after one success; if its final acknowledgement is lost, the sender must independently confirm the destination. Preserve partial files while resuming.

```sh
uv run python -m ops.checkpoint_sync serve --host 127.0.0.1 --port 53122 --file outputs/received-checkpoint.pt
uv run python -m ops.checkpoint_sync send --host 127.0.0.1 --port 53122 --file outputs/source-checkpoint.pt --simulate-drop-after 50000
```

The receiver requires hard-link support on its destination filesystem. A matching existing destination is accepted after size and hash verification; a different destination is rejected. Local tests cover actual interrupted byte-level resume, exact recovery, refusal to overwrite and corruption rejection. They do not establish unattended service resilience or security on an untrusted network.

An SSH-tunneled cross-laptop trial also resumed a 3,983,694-byte compressed
checkpoint at offset 50,000 after a deliberate disconnect. The published
4,379,345-byte checkpoint matched the source SHA-256; see
`checkpoint-tunnel-82994.json`. This is one controlled interruption, not a
test of arbitrary failures or automatic training restart.
