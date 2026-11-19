# Distributed inference: initial device checks

The networking prototype assigns fixed checkpoint/color/seed image jobs to
independent workers. The workers communicate through TCP on loopback and SSH
forwarding; this is task-parallel inference, not synchronized gradient
training. It combines existing mechanisms and is not a claim of a novel
networking algorithm. See `ops/README.md` for operation and security limits.

On the existing Windows laptop (RTX 3050 6 GB, PyTorch 2.9.1+cu128), an
SSH-tunneled worker loaded the completed low-balanced 500,000-update
checkpoint (SHA-256
`6ff581062b3410c31f72d0ce244db38dc2ca2f90c19614500299666458e77d93`).
One red and one blue image at seed 82990 were generated using its EMA sampler.
The first run took 53.87 seconds end to end and an immediately repeated run
44.27 seconds; both pairs of PNG hashes matched. This is a two-image,
single-worker repeatability check, not a speed benchmark. Loading, CUDA
initialization, generation, queuing and transfer are mixed in the timings;
there is no GPU-idle measurement. Cross-device equality is not implied.
The ignored local manifests have SHA-256
`ec819322a2e41f5f660803b78fa4119f8ee33107feb0ea7e9ed82b34cb87fe1f`
and `09043b031e504f2acc66154df106858ea37deb0edd6c047b219855a88716afa9`.
These first runs used the initial coordinator before resume support was added.
The revised coordinator then generated another red/blue pair (seed 82992)
through the same worker; a subsequent `--resume` verified and reused both
images without requesting them again. Unit tests exercise interrupted-job
recovery, a changed plan, and altered output. This live check does not itself
simulate a network failure midway through a job.

The same laptop also completed a separate three-update CUDA training smoke
on a newly generated 20-scenes-per-cell synthetic dataset (seed 82990),
with model seed 82991, batch 2, width 8 and four diffusion timesteps. The
checkpoint SHA-256 was
`8caf1b8e5c6a053f793bd328fb6916707bf9ef829b9ff350c7e0d9839963a5e5`.
This tiny run verifies plumbing only and does not show learning or close any
research gate. Some generic strings in the trainer's smoke report describe its
usual 200-image, 1,000-timestep configuration; the explicit configuration
and generated dataset above are the settings actually used here.

The new Windows laptop reported the same GPU model. Its isolated environment
is being provisioned; a completed two-worker GPU run and any speed comparison
must be documented separately after verification. A valid performance study
needs frozen jobs and model, independently repeated and interleaved single- and
two-worker runs, warmup, provenance checks and clear compute/network timing.

While provisioning, the new laptop received a separate CPU-rendered stream
over an SSH remote forward. For 300 scenes per setting, seed 82993, batches
2, 8 and 16 yielded respectively 555.6, 827.4 and 513.8 scenes/s (4.73, 7.05
and 4.38 Mb/s of PNG payload). The receiver timed decoding and waiting; the
measurements include producer rendering and encoding, not GPU training or raw
link bandwidth. These are single, sub-second sequential trials, so their
ordering is not a reliable batch-size comparison. Raw records are in
`ops/tunnel-second-laptop-82993.jsonl`; no private host address is included.

A separate SSH-tunneled checkpoint transfer to that laptop sent the same
4,379,345-byte checkpoint (3,983,694 gzip bytes). The sender deliberately
closed the first connection after 50,000 compressed bytes and reported resume
offsets `[0, 50000]`. The receiver independently reported SHA-256
`6ff581062b3410c31f72d0ce244db38dc2ca2f90c19614500299666458e77d93`
after publication, matching the source. This demonstrates one planned
interruption and byte-level recovery; it does not establish resilience to
power loss, adversarial peers, or every network failure mode.
