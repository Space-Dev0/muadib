# Plan: network-aware, experiment-preserving diffusion across laptops

Status: **proposal for discussion; not implemented or benchmarked.** This plan
extends the existing prototype and does not claim a new state-of-the-art result.
Device state and path observations are snapshots from 24 November 2026; they
can change. No credentials, public keys or private network addresses are stored
in this document.

## 1. Research idea

Build a small heterogeneous-device system that runs a **fixed, balanced
experiment plan** over a changing Tailscale network. Its scheduler chooses
where to execute each diffusion job using measured worker load and network
conditions; its durable job ledger tracks every requested condition, seed and
checkpoint. If a worker disconnects, the system retries the exact logical job
and preserves the original experimental quotas.

**Research question:** Under heterogeneous laptop speeds, changing network
paths and worker interruptions, can network-aware scheduling and resumable
work finish a prespecified multi-condition diffusion evaluation faster or more
reliably than simpler schedulers **without changing the planned sample
composition**?

This is a proposed systems contribution at the intersection of networking and
experimental validity. Adaptivity, retries, fairness constraints and edge
scheduling have prior art; their use here is a candidate research contribution,
not an established novelty claim. A literature comparison and ablation must
show what, if anything, is new and effective.

## 2. Inventory and known access

Tailscale reported the local Fedora coordinator and five online laptop peers:
Alter Ego (Windows), Prince's Fedora, NewAurora (Linux), Shreyas's Windows
laptop, and Nikhil's Windows laptop. Nishkarsh's Windows laptop and the phone
were offline. “Online” is Tailscale's last reported status, not proof that SSH,
the application or a GPU is available. This is five laptop peers **plus** the
local coordinator; use the coordinator only as a coordinator until its hardware
is verified. Device identities and the desired five-node configuration need to
be confirmed before the actual experiment.

- Alter Ego: SSH worked; one RTX 3050 Laptop GPU with 6,144 MiB was reported.
- Nikhil: the existing SSH endpoint reached a Linux WSL2 shell on the Windows
  host. NVIDIA tooling was not found on that endpoint; Windows-host and WSL GPU
  access remain unverified.
- Shreyas: Tailscale listed the device online but ping and SSH timed out during
  this check. Recheck when available.
- Prince and NewAurora: Tailscale ping responded, but their SSH host keys are not
  yet trusted locally. The operating-system usernames and GPU specifications
  must be verified; no password login or key installation was attempted.
- Local Fedora: `nvidia-smi` was unavailable in the current shell. It can act as
  a CPU coordinator unless hardware inspection shows otherwise.
- Individual direct Tailscale pings were observed to four peers; one did not
  respond. These are single snapshots, not full link-quality measurements.

Before trusting a new SSH host key, compare its fingerprint through a trusted
channel with the fingerprint shown locally on that laptop. After verification,
install a dedicated public SSH key and test access without interactive passwords.
Do not store passwords in files, code, logs, command history or the repository.
Use minimally scoped accounts and restrict services to trusted peers. A device
listed online does not have to participate if its owner has not agreed to the
experiment or it has incompatible resources.

## 3. Proposed system

```text
              Experiment owner / coordinator
         frozen plan + durable job ledger + dashboard
                       /   |   \\
          Tailscale + authenticated service channels
                   /       |       \\
       GPU worker A    GPU worker B    GPU worker C ...
       cached model    cached model    cached model
       inference       inference       inference
                   \       |       /
            verified outputs + timing records
```

Tailscale provides encrypted peer connectivity; actual paths can be direct or
relayed and can change. The application is a **coordinator/worker system over a
peer-capable network**. It is not a decentralized control plane merely because
Tailscale is a mesh. Initially, the coordinator is a single point of failure;
the ledger can support restart after its own process failure, but no automatic
leader failover or split-brain safety is claimed.

### 3.1 Fixed plan and scientific coverage

Freeze the logical job set before a run. A job key includes experiment ID,
model/checkpoint digest, generation condition, dataset/training-seed identity,
noise seed, sampler and evaluation protocol version. Set exact required counts
for each stratum, such as condition × color × training seed. Scheduling may
change the order or worker, **not silently change which samples are required**.

The existing fixed evaluation is a useful starting workload: four pilot
checkpoints × two colors × eight seeds = 64 image jobs. Their checkpoints and
training provenance must be supplied; the current two-color sampler does not
support arbitrary multi-predicate prompts. This exercise measures inference
and network behavior; it does not turn the pilot into a completed joint-exposure
intervention.

For each experiment, retain failed, retried, timed-out and unresolved jobs.
Report completed and missing counts by stratum. Do not discard difficult cases,
select only successful outputs, switch to a different checkpoint, or use the
unvalidated checker as a reward to cherry-pick outputs. The scheduling policy
may prioritize underrepresented **planned** strata, but it must eventually
complete the same frozen workload as its baselines. Adaptive sample allocation
is a separate exploratory study requiring a declared sampling and analysis
protocol.

### 3.2 Network- and worker-aware scheduler

Start with a transparent, measured predictor rather than an unvalidated
reinforcement-learning policy. For each candidate worker, estimate

```text
predicted completion time = queue wait + model/cold-start time
                          + expected compute time
                          + expected transfer time + communication delay
                          + expected failure/recovery cost.
```

Estimate components from bounded warm-up runs and rolling observations; include
uncertainty and avoid switching for tiny, noisy differences (hysteresis).
Account for checkpoint identity and whether that exact model is already cached.
Measure actual connection type and per-link behavior, not just the device's
Tailscale “online” flag. Enforce resource availability, a bounded queue, and
fixed stratum quotas. Treat predictive estimates as estimates, not guarantees.

Compare:

1. One-worker baseline on the fastest measured compatible laptop.
2. Static round-robin over available workers.
3. Least-outstanding-work or least-queue baseline.
4. Compute-only scheduling, ignoring network state.
5. Compute + network scheduling.
6. Proposed compute + network + failure cost + fixed-plan coverage policy.

Use exactly the same jobs, models, quality settings and completion criteria in
each comparison. Include ablations for coverage priority, network telemetry,
and retry/checkpoint overhead. Fault injection and device churn should be
predeclared and reproducible.

### 3.3 Fault tolerance and integrity

Extend the existing coordinator, which already has a frozen job plan, validated
PNG outputs, retry and resume, rather than starting from scratch. Persist job
states and attempts: planned → leased/running → completed or failed. Use leases
and retry policies, checkpoint/worker fingerprints, checksums and atomic result
publication. Multiple attempts can occur; ensure a logical job has at most one
accepted published result while retaining attempt records. **Do not claim
exactly-once computation.** Record timeouts and failed attempts rather than
silently omitting them from timing or quality summaries.

Preserve the existing resumable compressed checkpoint transfer as a separate
artifact-transport path. SHA-256 checks byte consistency against a trusted
expected digest; it does not authenticate a peer. SSH or mutually authenticated
TLS and explicit authorization are needed for peer identity and transport
security. Never accept an untrusted PyTorch checkpoint or silently trust a new
SSH host key.

### 3.4 Optional ambitious extension: migrate a partially denoised image

After the whole-image baseline works, implement an experimental **one-time
mid-sampling handoff** between compatible workers. The first worker performs a
prefix of reverse steps; the second resumes the same job from a durable sampler
state if a planned handoff or failure occurs. Save the current tensor, next
step index, exact scheduler parameters, condition, checkpoint identity,
precision and library/runtime metadata, and the complete random state required
by the sampler. Record the transfer and any redo work.

Prove same-device interrupted-versus-uninterrupted equivalence first. A
cross-device continuation can differ numerically or use incompatible RNG state;
do not promise bitwise equality or claim quality preservation without tests.
If a common per-step random stream is introduced, it changes the sampler
implementation and must also be used in all comparison baselines. Evaluate the
whole output set with a validated evaluator or blinded review; the present
checker alone is insufficient.

At 64×64 RGB float32, the current image tensor is 64 × 64 × 3 × 4 = 49,152
bytes (48 KiB), before framing and metadata. A single handoff can therefore be
small, but transferring an image-sized tensor to a remote stage and back at
**every one of 1,000 steps** would move roughly 93.75 MiB per image in raw
float32 tensors alone, plus at least 1,000 network round trips and any
intermediate activations. Actual traffic depends on the partition and protocol;
this back-of-the-envelope figure is not a benchmark. Small per-step compute can
make splitting slower. Compare against whole-image routing rather than assuming
a speedup.

A dynamic split-timestep or mixed-model relay would be a separate research
extension with compatibility, trajectory and output-quality risks; prior work
already studies such relay inference. Avoid swapping unrelated checkpoints or
averaging model weights without a validated method.

## 4. Network topology, Meshery and what is actually novel

First measure the physical/logical testbed: a timestamped five-node selection,
OS, GPU/VRAM and driver/runtime, checkpoint cache state, pairwise latency,
throughput and direct/relay status. A five-node full pairwise matrix has 10
unordered pairs (or 20 directional measurements); the coordinator may be an
additional node. Tailscale link state can change, so label snapshots by time.

A peer-capable overlay makes direct worker-to-worker transfer or a checkpoint
handoff possible, but each proposed application hop must actually be measured;
a logical worker mesh is not the same as changing Tailscale routing. An optional
visual dashboard can show peers, connections, queues, job states and recovery.
It should distinguish directly observed measurements from estimates.

Meshery is a cloud-native management and visualization platform, not the
scheduler, VPN or proof of novelty. Its FAQ says configuration/lifecycle
management features require Kubernetes, although it offers some operations
outside Kubernetes. Introducing Kubernetes/WSL/containers on mixed Windows
and Linux laptops adds setup and failure modes. Keep the first implementation
as Python services over SSH/Tailscale; evaluate Meshery as a later optional
visualization/deployment experiment, only if its added value is measurable.

## 5. Experiments and evidence

The accompanying [measurement specification](network-metrics-plan.md) defines latency, RTT variation, goodput, job timing, recovery, eBPF attribution, scientific coverage and the proposed reporting protocol.

Freeze one workload manifest and preserve all outputs and attempt records.
Use repeated, interleaved paired runs where practical, with warm-up and both
cold-cache and warm-cache cases explicitly labeled. Measure:

- Completion rate and missing jobs **by condition/stratum**.
- End-to-end makespan from initial submission to final completion, including
  retries, cold starts and failed attempts; also successful throughput.
- Per-job latency distributions (median, p95 when sample size supports it),
  queue time, compute time, network time, bytes, retry count and recovery time.
- Worker availability, GPU utilization where meaningful, checkpoint cache hits,
  connection path (direct/relay/unknown) and telemetry overhead.
- Output integrity and, separately, image quality/semantic validity with an
  appropriately validated checker and/or blinded human evaluation. Never treat
  hashes or checker abstention as proof of quality.
- Condition coverage and distribution of missing/failed work across strata.
  Complete the fixed plan when comparing outputs; report censored or unresolved
  runs explicitly rather than reporting only successful jobs.

Test at least: all workers stable, one slow worker, one mid-job disconnect,
worker return, bandwidth/latency degradation, and cached versus missing model.
Compare one-, two-, three-, four- and five-worker runs only when that many
compatible workers are actually available. Report hardware, duration, seeds,
uncertainty and all failed runs. Controlled fault injection should target our
own test processes or a test network namespace, not disrupt a teammate's
unrelated work.

Meaningful results are improvements against the strongest appropriate baseline
under the same workload, with overhead and regressions included. No particular
speedup is assumed. If the adaptive scheduler loses to round-robin, report it;
that result also informs when a more complicated scheduler is worthwhile.

## 6. Delivery phases

1. **Inventory and trust:** verify host fingerprints and owners; establish SSH
   keys; check GPU model/VRAM, OS, compatible runtime, sleep and storage.
   Measure baseline direct/relay connectivity. Do not start training.
2. **Reproduce the current system:** run a tiny two-worker image workload and
   verify manifests, retries and existing checkpoint transfer before changing it.
3. **Fixed-plan five-device runner:** add durable per-job/attempt records,
   exact model provenance, dashboard-ready telemetry and complete-stratum
   accounting. Test both CPU fake workers and available real GPU workers.
4. **Schedulers and networking study:** implement the baselines and proposed
   network/worker-aware policy, then run paired multi-device and fault tests.
5. **Optional mid-denoising recovery:** first validate uninterrupted versus
   interrupted output on one device, then test compatible cross-device handoff;
   keep this separate from baseline claims.
6. **Research write-up and presentation:** cite adjacent work; add topology,
   network observations, reproducible benchmark protocol, baselines, actual
   results and limitations. No fabricated results or claim of novelty without
   the comparison and evidence.

## 7. Relevant research and technical references

- [RISE: Relay Inference and Online Scheduling for Efficient Edge-Device Collaborative Diffusion Model Services (2026)](https://arxiv.org/abs/2606.17378):
  combines staged model relay with context-aware scheduling. Directly relevant
  prior art; our proposed fixed-plan coverage and fault recovery must be
  distinguished empirically rather than called the first network-aware
  diffusion scheduler.
- [EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling (2025)](https://arxiv.org/abs/2507.10026):
  considers heterogeneous edge servers, QoS and AIGC scheduling. Useful
  reference for related work and baselines, not a claim that we implement its
  reinforcement-learning policy.
- [Decentralized Diffusion Models (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/McAllister_Decentralized_Diffusion_Models_CVPR_2025_paper.html):
  trains isolated experts and combines them at inference. Our initial system
  schedules independent jobs; it does not reproduce or inherit the paper's
  ensemble theory.
- [Heterogeneous Decentralized Diffusion Models (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Heterogeneous_Decentralized_Diffusion_Models_CVPR_2026_paper.html):
  studies heterogeneous experts with substantially larger resource assumptions;
  it is relevant context, not an off-the-shelf 6-GB-laptop recipe.
- [DistriFusion (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_DistriFusion_Distributed_Parallel_Inference_for_High-Resolution_Diffusion_Models_CVPR_2024_paper.pdf):
  distributes high-resolution diffusion inference and discusses communication
  costs. Its high-bandwidth multi-GPU results should not be projected onto a
  small 64×64 model over laptops.
- [Tailscale connection types](https://tailscale.com/docs/reference/connection-types):
  explains direct, DERP-relayed and peer-relay paths and how to identify them.
- [Meshery FAQ](https://docs.meshery.io/project/faq/): describes its scope and
  Kubernetes dependence for configuration/lifecycle-management features.
