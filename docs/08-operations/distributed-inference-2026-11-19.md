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
