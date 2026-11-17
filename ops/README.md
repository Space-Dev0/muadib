# Networking prototypes

These are experimental protocols, not a completed Stage 0 result. They do not encrypt or authenticate peers: use only over an authenticated tunnel or isolated trusted network. SHA-256 detects corruption, not malicious modification by an attacker. Do not expose prototype ports publicly.

## Unconditional scene streaming

`scene_stream` sends batches of CPU-rendered 64×64 PNG images and JSON annotations. It is separate from the training experiment; the demonstrated distribution is unconditional.

```sh
uv run python -m ops.scene_stream serve --host 127.0.0.1 --port 53121 --count 300 --batch-size 8
uv run python -m ops.scene_stream receive --host 127.0.0.1 --port 53121 --batch-size 8 --metrics outputs/stream-metrics.jsonl
```

Start the producer in one terminal, then the receiver in another. On the receiver, `--batch-size` labels the report and must match the producer. `receive_decode_seconds` includes blocking, frame parsing and PNG decoding; it is not GPU idle time or pure link latency. Payload throughput counts PNG bytes only; it excludes JSON and TCP/SSH overhead and is not a measurement of physical link capacity.

`local-benchmark.json` records a short localhost run (300 scenes each for batch sizes 2, 8 and 16). Reproduce with `uv run python -m ops.benchmark --count 300 --output outputs/network-local.json`.

`tunnel-benchmark.jsonl` records one short real inter-machine run in the direction **Linux producer → Windows laptop receiver**, using SSH remote port forwarding over the existing private connection; 300 scenes, seeds 90000–90299, per batch size. Results are 582.0/801.1/925.7 scenes/s and 4.91/6.76/7.81 Mb/s PNG payload for batches 2/8/16 respectively. The receiver's `host` field is loopback on its side of the SSH tunnel, not the remote machine address. These are three individual sub-second trials, not the requested 15-minute LAN benchmark, no error bars or evidence of stable differences. The producer uses the unconditional renderer; no model or GPU participated.

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

`checkpoint_sync` compresses a stable source with gzip, validates compressed and decompressed SHA-256, resumes interrupted compressed transfer from its partial byte length, and publishes only after validation. Existing destination files are never overwritten. It does not perform delta encoding or automatically resume training. The receiver exits after one success; if its final acknowledgement is lost, the sender must independently confirm the destination. Preserve partial files while resuming.

```sh
uv run python -m ops.checkpoint_sync serve --host 127.0.0.1 --port 53122 --file outputs/received-checkpoint.pt
uv run python -m ops.checkpoint_sync send --host 127.0.0.1 --port 53122 --file outputs/source-checkpoint.pt --simulate-drop-after 50000
```

The receiver requires hard-link support on its destination filesystem. A matching existing destination is accepted after size and hash verification; a different destination is rejected. Local tests cover actual interrupted byte-level resume, exact recovery, refusal to overwrite and corruption rejection. They do not establish unattended service resilience or security on an untrusted network.
