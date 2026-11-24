# Measurement plan for the multi-device diffusion experiment

Status: **measurement specification, not experimental results**. This document
accompanies the [multi-device research plan](five-device-network-research-plan.md).
The actual device inventory, access, workload, measurement tools, clock accuracy
and available GPU features must be verified before the experiment.

## 1. What are we trying to measure?

We want to find out whether a network-aware coordinator can complete a **fixed,
balanced set of diffusion-generation jobs** reliably and efficiently across
heterogeneous laptops when network paths, compute speeds and availability vary.
The system must not silently omit difficult conditions or count duplicate
attempts as additional independent samples.

There are five groups of measurements:

1. Network health: latency, delay variation, loss, goodput and retransmissions.
2. End-to-end system performance: job time, throughput and experiment duration.
3. Reliability: failures, retries, recovery, integrity and workload completion.
4. Resources and measurement overhead: GPU, CPU, memory, energy when available.
5. Scientific validity: whether the exact planned conditions and seeds were
   retained, and whether generated outputs satisfy an appropriately evaluated
   rule.

Measure **the same definitions** for each scheduling policy. Metrics obtained
by different tools or at different layers must not be merged without labeling
how they were measured.

## 2. Freeze the experiment before measuring

Record a versioned workload manifest before each paired comparison:

- run ID; logical job ID; stratum (condition, target color/rule, dataset seed,
  training seed); generation noise seed; and expected number of jobs;
- checkpoint and configuration hashes, sampler version, timesteps, guidance,
  precision and model/software versions;
- which workers are included, operating system, GPU and VRAM, driver/runtime,
  power mode, checkpoint cache state, and coordinator location;
- scheduler policy and parameters, timeouts, retry rules, probe interval,
  failure schedule, duration, warm-up and cool-down;
- measurement-layer definitions, timestamp sources, exclusions and analysis
  plan.

The existing pilot's 4 conditions × 2 colors × 8 noise seeds gives 64 logical
jobs, provided all four compatible checkpoints are available. The existing
sampler is color-conditioned; it does not yet implement arbitrary relational
prompts. The pilot workload is useful for testing the network system, but it is
not the planned joint-exposure intervention.

A logical job can have multiple **attempts**. Give every attempt its own ID,
worker, start/end times and status. Commit at most one verified result for a
logical job and retain all attempts. A retry is not a new scientific sample.

## 3. Core network metrics

For each directed source–destination pair and measurement interval, record the
protocol, application path, direction, start/end time, sample count and whether
the path was observed as direct, DERP relay, peer relay or unknown. On five
selected peers there are 10 unordered pairs, or 20 directional measurements;
include the coordinator separately if it is a sixth node.

| Metric | Definition and unit | How to collect / report |
|---|---|---|
| **Probe round-trip latency (RTT)** | Time from a probe's send to its corresponding response at the same source, in milliseconds. | A bounded application echo or an explicitly labeled ping; report median, p95, min, max, count and timeouts. Do not call it one-way latency. |
| **TCP socket RTT estimate** | Kernel's estimate for an identified TCP connection, in ms, where available. | Linux socket/eBPF telemetry for the specific SSH or application socket; record estimator, collection interval and source. It is not necessarily the same as echo RTT. |
| **RTT variation** | For consecutive successful, regularly spaced probes, `d_i = abs(RTT_i − RTT_(i−1))`, in ms. | Report median/p95 of `d_i`, probe spacing and sample count. Call it **RTT variation**, not RFC one-way IP packet delay variation. |
| **Probe timeout rate** | `number of probes with no reply before the declared timeout / probes sent × 100%`. | Report the timeout threshold, interval and both numerator and denominator; it is a probe-outcome metric, not necessarily packet loss. |
| **Packet loss, if instrumented** | `missing sequence-numbered packets / packets sent × 100%` in a defined controlled datagram test. | Requires cooperating endpoints, sequence numbers, a loss deadline and a description of duplicates/reordering; distinguish directions. TCP retransmissions alone are not packet-loss percentage. |
| **Application payload goodput** | `8 × unique useful payload bytes delivered / elapsed seconds`, in bit/s or Mb/s. | Count received PNG, metadata or checkpoint bytes separately; specify start/end boundaries and whether retries are excluded from the useful numerator. |
| **Bytes sent, received and retried** | Measured byte totals, in bytes or MiB, for a declared layer and direction. | Application framing counters plus optional socket/interface counters; separate first attempts from retransfers. Do not add counts from multiple layers as though independent. |
| **TCP retransmission events** | Number of observed retransmission events for identified TCP connection(s) during an interval. | Optional Linux eBPF/kernel metric; record hook, flow identity, direction, interval and count. Retransmission events are not lost packets, unique payload bytes or a loss rate. |
| **TCP connection setup time** | Same-host elapsed time from connection attempt to successful connection establishment, in ms. | Application/socket events; separately report SSH session/tunnel creation or reuse. A timeout is not a successful latency sample. |
| **Path/connectivity changes** | Observed direct/DERP/peer-relay/unknown path, connection interruptions and transitions over time. | Tailscale status or explicitly tagged probes; record timestamps, observation age and unknown values. Do not claim a precise transition time between snapshots. |

### “Jitter” and one-way delay

In a presentation, **jitter** can mean changing delay, but it has multiple
formal definitions. Our default number is the explicitly defined **RTT
variation** above. [RFC 3393](https://www.rfc-editor.org/rfc/rfc3393.html)
defines IP packet delay variation using selected **one-way** delay differences;
our `abs` difference of round-trip probes is not that RFC metric.

Do not calculate one-way latency by subtracting timestamps from different
machines unless the clocks' synchronization method, offset, drift and
uncertainty are measured and recorded. `RTT / 2` is not guaranteed to be
one-way latency because paths and processing can be asymmetric. Use
same-machine monotonic durations for the primary latency and recovery metrics.

The current tools use different explicitly labeled percentile conventions.
`ops.network_probe` uses nearest rank, including the lower middle value for an
even sample; its echo RTT starts after connection establishment and stops when
the echoed bytes arrive. `ops.benchmark_report` averages the two middle values
for an even-sample median and linearly interpolates p95 at rank `(n-1)*0.95`.
Its successful request interval starts on entry to the coordinator's request
function and ends after response transfer, SHA-256 and structural PNG checks;
it excludes prior queue wait and subsequent result decoding and publication.
These intervals describe different operations and should not be compared as
though they were measurements of the same latency.

### Link goodput versus job throughput

A link test sends controlled data to characterize a path; its Mb/s is not
necessarily an image-generation rate or raw physical line capacity. A job
benchmark reports completed images per second. PNG payload Mb/s excludes
metadata and transport overhead; kernel interface counters may include tunnel,
control and unrelated traffic. Name the layer and measurement window for every
number. [RFC 6349](https://www.rfc-editor.org/rfc/rfc6349.html) provides TCP
throughput-testing context but is not itself a measurement of this experiment.

## 4. Job and experiment performance

All time intervals below use a monotonic clock at the component that records
both ends. Cross-machine timestamps may be used for correlation only with
recorded synchronization uncertainty. Use event IDs rather than assuming that
unsynchronized clock differences yield one-way network delay.

| Metric | Exact operational definition | Report |
|---|---|---|
| **Run makespan** | Coordinator time from submitting the frozen batch until the last required job is validly committed, including queueing, failures and retries. | Seconds; if any required job remains missing at the deadline, mark the run incomplete and report elapsed time and completion fraction rather than pretending it finished. |
| **Completion fraction** | `number of distinct required jobs with one valid committed output / number of required jobs`. | Overall and for every fixed stratum, including missing and permanently failed jobs. |
| **Goodput: jobs/s** | `number of distinct valid committed jobs / coordinator elapsed time`. | Use the declared run window, include failed attempts and recovery time in elapsed time, and report incomplete runs separately. |
| **Job response latency** | Coordinator time from logical job submission until its verified result is committed. | Median and p95 with completed/total counts; timeouts and unfinished jobs remain visible rather than being silently excluded. |
| **Queue wait** | Coordinator duration from job eligibility/enqueue until the chosen dispatch attempt begins. | Per job and attempt; distinguish retry queueing and worker-side queues if present. |
| **Worker execution time** | Worker monotonic duration around the defined sampler execution. | Include or separate checkpoint loading and initialization explicitly; do not subtract timestamps from different machines. |
| **Checkpoint cold-start time** | Time to locate/transfer/load and initialize the specified checkpoint under a defined cold-cache protocol. | Report cache hit/miss, file size, transfer and load components when separately measured; warm and cold results must not be mixed. |
| **Result publication time** | Coordinator duration from receiving a full output to completing its integrity checks and durable/atomic publication, as implemented. | Report validation method and filesystem semantics. Atomic publication alone is not a power-loss durability guarantee. |
| **Speedup** | `baseline makespan / proposed makespan`, only for completed runs with the same fixed workload and scope. | Use the fastest verified single-worker baseline and the same warm/cold setting. Do not infer speedup from unmatched jobs or durations. |
| **Scaling efficiency** | `speedup / number of participating workers`, for a like-for-like strong-scaling comparison. | Also report worker hardware heterogeneity, coordinator inclusion and incomplete runs; the percentage is not GPU utilization. |

**Important:** Sum of per-stage timings may not equal wall-clock job latency
when network transfer, computation and queuing overlap. Label an estimate as an
estimate. Separate queue wait from actual GPU execution if GPU timing is
available; wall time for a remote request is not pure GPU idle time.

## 5. Reliability and fault recovery

Fault experiments must use controlled test workloads on consenting devices.
Record injection start/end and exact type. Avoid disrupting teammates' other
processes. Initial proposed fault scenarios: a stopped worker process, delayed
worker, interrupted connection, worker return, and model-cache miss.

| Metric | Definition / reporting rule |
|---|---|
| **Attempt failure rate** | Failed attempts / total attempts, with timeout, transport, worker and validation failure types reported separately. |
| **Logical-job failure / missing fraction** | Required jobs not validly completed by the declared deadline / required jobs; include final reason and stratum. |
| **Retry count and extra work** | Attempts beyond the first, duplicated compute time where observable, retransferred bytes and repeated checkpoint loads. |
| **Failure-detection delay** | Coordinator observation time of the declared injected fault minus its injection time, when both are on the same clock or synchronization uncertainty is known. |
| **Recovery-to-commit time** | Coordinator duration from fault detection to the first valid commit of the affected job; also report fault-to-commit if the injection time is comparable. |
| **Lost work / redo** | Actual previously completed but repeated work, measured as time, steps or bytes with defined boundaries. Do not infer progress from timestamps alone. |
| **Resume offset and transfer integrity** | Offset in compressed bytes when resuming; bytes sent before/after interruption; compare compressed and decompressed digests with trusted expected values. |
| **Result integrity failures** | Counts of missing, malformed, wrong-configuration or hash-mismatched results; retain failed records. |
| **Duplicate execution and publication** | Number of jobs with more than one execution attempt and number of duplicate accepted publications. Goal: one accepted result per logical job; no claim of exactly-once execution. |
| **Coordinator recovery** | For a controlled coordinator-process restart, ledger recovery time and number of jobs in each state. | Local durable restart is separate from automatic coordinator failover; do not claim leader election without implementing and testing it. |

If a run never completes, preserve it as incomplete. Do not compute recovery
latency only over successful recoveries without reporting the failure and
censoring counts. Report the deadline and observation window.

## 6. Worker health and overhead

| Metric | Definition / source | Caveat |
|---|---|---|
| **GPU utilization** | NVIDIA/NVML-reported utilization sampled at declared intervals, per device. | A sampled busy percentage is not achieved compute throughput, job progress or a universal cross-GPU performance measure. |
| **GPU memory** | Used and total VRAM, plus per-process usage if available. | Shared processes and allocation caching can affect readings. |
| **CPU and RAM** | Per-process and host utilization/RSS, sampled at declared intervals. | Record core count, competing load and platform-specific definitions. |
| **GPU model/driver/runtime** | Device model, VRAM, driver, CUDA/PyTorch version, OS and power mode. | A machine reported online may not have a usable GPU in the actual worker environment. |
| **Temperature, clocks and throttling** | GPU temperature, reported clock and throttling indicators where exposed. | Availability and meaning differ by GPU/OS; correlate with time without claiming causation. |
| **Power / energy, optional** | GPU power samples in watts and estimated energy by integrating samples over time, in joules or Wh. | Only where exposed; a GPU reading is not whole-laptop energy. Record sampling rate, measurement scope, missing values and integration method. |
| **Telemetry overhead** | Difference in resource use and run performance between matched telemetry-off and telemetry-on runs. | Run both round-robin and adaptive policies with comparable measurement settings; include event loss and collector CPU/memory. |
| **Sleep / availability** | Worker heartbeat status and logged worker availability windows. | A missing heartbeat is an observation, not proof of a particular cause such as sleep. |

## 7. Scientific coverage and output validity

Networking must not change the scientific question while improving a system
metric. Predeclare the target number of logical jobs for each rule/condition,
model and seed stratum. Report:

1. Planned, submitted, completed, missing and failed logical-job counts per
   stratum and per worker; identify any worker/condition confounding.
2. Checkpoint, condition, seed, sampler and code/runtime provenance for each
   committed result. Reject incompatible models rather than silently switching.
3. Number of duplicate attempts and committed outputs; retain the complete
   attempt log.
4. Output-file validity and digest checks separately from image semantics. A
   matching hash confirms the checked bytes, not that the image is correct.
5. For output quality, use a predeclared validated checker and/or independent,
   blinded human evaluation when feasible. Report yes/no/uncertain, checker
   abstention, coverage and the applicable denominators. The existing checker
   is not a validated source of universal image-quality labels.
6. Compare the *same complete planned jobs* across policies. If an attempt is
   rerun on different hardware, record that worker change and assess numerical
   differences; identical seeds do not guarantee identical outputs across
   devices or runtimes.

Scheduling priority can change job order but must not silently change sample
quotas, timesteps, checkpoint versions or quality settings. If a deadline
prevents completion, report condition-specific censoring. Do not compare only
outputs that happened to finish quickly.

## 8. eBPF: what it can and cannot contribute

On supported Linux machines, a passive eBPF observer may collect selected TCP
connection/retransmission and kernel timing counters. Linux eBPF loading needs
suitable kernel support and privileges. Use limited scope, filtering,
aggregation, documented hooks and bounded storage. Record collector errors and
lost/dropped telemetry events. Do not collect payload contents.

The local Fedora check found kernel BTF, `bpftool` and `clang`, but unprivileged
BPF loading was disabled. That does **not** establish that any proposed probe
can be loaded or that another machine supports the same hooks. Check each
kernel and privilege boundary before using eBPF. Native Windows eBPF and Linux
WSL have different support and visibility; begin with application-level
telemetry on devices that cannot run the Linux observer.

**Attribution is a separate engineering task.** Application jobs can use local
sockets forwarded through an SSH process, and several jobs may share one
transport connection. A TCP retransmission event may occur outside the original
process context. Associate flows using socket/connection metadata, time windows
and application job IDs where possible; do not claim that every byte or
retransmission belongs to one job without validating that mapping. Tailscale's
outer encrypted tunnel and the inner TCP connection are different layers.

Do not describe eBPF as directly measuring GPU utilization, one-way latency,
physical bandwidth, UDP packet loss or output quality. Record unavailable
metrics as **unavailable**, not zero. An application-only telemetry policy must
remain usable when probes are missing or stale.

## 9. Experimental comparisons and practical protocol

### Policies

Compare the same frozen workload under:

1. Strongest verified single-worker baseline.
2. Round-robin over available workers.
3. Least-queued-work scheduling.
4. Compute-aware scheduling.
5. Compute + application-level network measurements.
6. Compute + network + fault-aware scheduling, with a fixed-plan coverage
   constraint; compare application-only versus eBPF-assisted telemetry where
   Linux coverage permits.

Round-robin with telemetry off/on estimates observer overhead. Comparing
adaptive scheduling with application versus eBPF telemetry estimates the
incremental value of eBPF, not merely the value of adding any monitoring.
Retain identical jobs, checkpoints, sampler settings, fault scenarios, deadlines
and quality/evaluation rules across policies.

### Repetitions and conditions

Record idle and loaded network states; warm and cold checkpoint-cache runs;
1, 2, 3, 4 and 5 compatible worker configurations where available; and
controlled slow-worker, disconnect and return scenarios. Pair and interleave
policies under comparable conditions where feasible. A suggested starting
point is at least five paired complete runs per scenario after warm-up, with
more runs where time permits. This is a practical minimum proposal, **not** a
claim that five runs give adequate statistical power or stable p95 estimates.
Record all samples and determine an appropriate run count before final claims.

Predefine probe frequency (e.g. a bounded 1-second diagnostic echo while a test
is active), duration, timeout, payload, ports and rate limits; measure probe
and collector overhead. Use a separate controlled sustained throughput test
only with the participants' agreement and a stated duration. Avoid saturating
shared links during teammates' work. Record direct/relay/unknown path over time;
do not infer it from an old `tailscale status` snapshot.

### Statistical reporting

The primary independent unit for scheduler comparisons is the **run** or other
explicitly randomized experimental unit, not every correlated image produced by
one model. Pair by the same frozen workload/fault scenario where possible.
Report per-run values, number of runs, effect sizes and uncertainty; if using a
bootstrap or other interval, resample at the run or appropriate cluster level,
not as though every image were an independent training run. Tail percentiles
from small counts should be labeled exploratory and accompanied by sample
counts. Report failed, censored and incomplete runs alongside completed ones.

The 64 images are not 64 independent model-training replications. Training
seed, dataset seed and inference-noise seed are distinct sources of variation.

## 10. Suggested figures and tables for the final report

| Figure / table | Axes or fields | Main point |
|---|---|---|
| Topology and device inventory | Node ID, OS, GPU/VRAM, worker/coordinator role; no private addresses or credentials | What was actually tested. |
| Pairwise RTT heatmap | Source vs destination; median RTT; path type; missing/unknown cells | Network heterogeneity. |
| RTT-variation and timeout plot | Time vs explicitly defined RTT variation and probe timeouts | Changing network conditions; no unlabeled “jitter.” |
| Payload-goodput plot | Batch/payload size vs application Mb/s, with direction, repetitions and uncertainty | Transfer behavior at the measured layer. |
| Makespan vs policy and worker count | Same frozen workload, repeated runs; show incomplete runs separately | Whether scheduling or additional workers help. |
| Per-job latency distribution | Median/p95 and raw samples, with timeout/censored counts | Typical and tail behavior. |
| Fault timeline | Injection, detection, retry, worker return and first valid commit | Recovery mechanism and delay. |
| Condition-coverage table | Planned/completed/missing/failed/duplicate by stratum | Whether the scientific evaluation remained balanced. |
| eBPF ablation | App-only vs eBPF-assisted policy, plus telemetry off/on overhead | Whether kernel telemetry adds value beyond ordinary timing. |
| Resources and integrity | GPU/CPU/memory, collector overhead, checksum outcomes and unavailable fields | Cost and operational limits. |

## 11. Existing numbers are preliminary

The existing scene-stream prototype records application-level elapsed time and
PNG payload bytes. A longer localhost run measured about 1,082 / 1,061 / 1,060
scenes/s for batch sizes 2 / 8 / 16. Three separate short cross-laptop trials
measured about 582 / 801 / 926 scenes/s and 4.91 / 6.76 / 7.81 Mb/s PNG
payload. These are different workloads and durations; they include rendering
and encoding and are not GPU throughput, raw link capacity or a controlled
scheduler comparison. Do not present them as the results of the proposed
five-device/eBPF experiment. See [networking prototype records](../../ops/README.md).

## 12. References

- [RFC 2681: A Round-trip Delay Metric for IPPM](https://www.rfc-editor.org/rfc/rfc2681.html).
- [RFC 3393: IP Packet Delay Variation Metric for IP Performance Metrics](https://www.rfc-editor.org/rfc/rfc3393.html).
- [RFC 6349: Framework for TCP Throughput Testing](https://www.rfc-editor.org/rfc/rfc6349.html).
- [Tailscale connection types](https://tailscale.com/docs/reference/connection-types).
- [BCC tools for Linux BPF networking measurements](https://github.com/iovisor/bcc).
- [eBPF for Windows: supported model and platform differences](https://github.com/microsoft/ebpf-for-windows).
