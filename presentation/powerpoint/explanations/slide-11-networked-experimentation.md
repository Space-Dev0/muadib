# Slide 11 — Networked experimentation pipeline

This guide explains the networking slide from first principles, then describes what was implemented and measured. It distinguishes the CPU scene stream, checkpoint transfer, and two-worker inference, because they are related components but are not one benchmark. A concise audience script appears at the end.

## 1. What problem does networking solve here?

The project has scene generation, model training, checkpoints and model inference. These tasks may run on different machines. Networking allows a process on one machine to send data to another and lets a coordinator manage independent work across available devices.

The project prototypes three complementary capabilities:

1. **Scene streaming:** send rendered images and JSON annotations to a receiver.
2. **Resumable checkpoint transfer:** continue a partially received compressed checkpoint and verify the result.
3. **Task-parallel inference:** assign independent image-generation jobs to GPU workers and record their outputs.

These are engineering integrations of established ideas. They are not a new TCP protocol, proof of a faster diffusion algorithm, or proof of synchronized multi-GPU training.

## 2. The overall picture

```text
CPU scene producer
  64 × 64 PNG + JSON annotations
           |
        framed TCP
           |
   SSH port forwarding
  (possibly over a VPN)
           |
       receiver

Coordinator --> worker 1 --> generated image + metadata
            --> worker 2 --> generated image + metadata

Checkpoint sender --> interrupted transfer --> resume --> verify --> publish
```

The CPU scene-stream benchmark and the GPU inference check are separate experiments. The diagram shows the project components conceptually; it does not mean that the measured scene stream fed a cross-machine GPU training run.

## 3. Data plane: what is sent?

The scene producer constructs a 64 × 64 synthetic image and a structured scene description. The image is encoded as **PNG**; the annotations are represented in **JSON**. Depending on the protocol, annotations describe objects, attributes, and other metadata needed by the receiving workflow.

A **producer** creates and sends data. A **consumer** or **receiver** reads and decodes it. Separating them allows scene generation and consumption to occur in different processes or on different machines. It can also make the data path easier to inspect and measure.

The unconditional scene-stream benchmark uses CPU rendering. A separate experimental request-driven conditional adapter connects to a pilot training interface and checks rule/source provenance, image checksums, dimensions and returned annotation satisfaction. Its recorded parity and resume check was a tiny same-host CPU test. It does not establish a cross-machine GPU-training speedup or general equivalence across environments.

## 4. What is TCP and why is framing needed?

**TCP** is a reliable, ordered byte-stream transport. It gives an application a stream of bytes; it does not automatically preserve the sender's logical “one image,” “one annotation” or “one batch” message boundaries. The receiver may obtain a message in several reads, or multiple messages in one read.

The prototype uses application-level **framing** so the receiver can parse intended pieces of a message and know where the payload ends. Framing is a way to structure bytes on top of TCP, not a claim that TCP itself understands images or JSON.

A receiver typically needs to parse the frame, read the required amount of data, decode the PNG, process annotations and report metrics. Each stage contributes work or waiting time.

## 5. What are SSH forwarding and Tailscale doing?

**SSH** can authenticate a remote connection and encrypt traffic carried through it. **SSH port forwarding** maps a port at one end of that authenticated connection to a port at the other, so an application can communicate through the SSH tunnel while connecting to a local endpoint.

A VPN such as **Tailscale** can provide a private network path between the machines. SSH forwarding can then operate across that path. Tailscale supplies connectivity; it does not by itself turn our application-level checksums into peer authentication. The prototype's own TCP messages and SHA-256 checks should be used with trusted peers or an authenticated tunnel.

The slide says SSH forwarding *works across VPNs such as Tailscale*. Treat that as an available deployment arrangement, not evidence that every reported benchmark measured VPN overhead or that the connection was always relayed or always direct.

## 6. What does the throughput graph show?

The graph compares recorded CPU scene-stream throughput for batch settings 2, 8 and 16. The localhost run used **300,000 scenes for each setting**, or 900,000 scenes total. The shown cross-laptop SSH-tunnel experiment used **300 scenes per setting**.

| Scenes per batch | Localhost scenes/s | Cross-laptop scenes/s | Cross-laptop PNG payload Mb/s |
|---:|---:|---:|---:|
| 2 | 1,082 | 582 | 4.91 |
| 8 | 1,061 | 801 | 6.76 |
| 16 | 1,060 | 926 | 7.81 |

The local figures are rounded from approximately 1,082.10, 1,061.01 and 1,059.87 scenes/s. The remote figures are rounded from 582.0, 801.1 and 925.7 scenes/s. The local experiment took about 843 seconds of measured transfer time across the three settings, around 14.1 minutes in total. The remote points are individual short, sub-second trials; they do not establish stable rankings or confidence intervals.

### Throughput and payload rate

If `N` scenes are received over an elapsed time `T`, application throughput is

\[
 \text{scene throughput}=\frac{N}{T}\quad\text{scenes per second}.
\]

If the encoded PNG payload sizes are `b_i` bytes, the reported PNG payload rate is

\[
 \text{PNG payload rate}=\frac{8\sum_i b_i}{T}\quad\text{bits per second}.
\]

`Mb/s` means **megabits per second**, not megabytes per second. The payload measure counts PNG bytes; it excludes JSON and TCP/SSH overhead. The application measurements include rendering, encoding and receiving work in their measured interval. They are not pure physical link capacity, GPU utilization, GPU idle time, or diffusion-training throughput.

Batch size here is the number of scenes per protocol batch. It should not be confused automatically with the training batch size on slide 8.

### How should we interpret the bars?

The graph shows that data moved through the prototype under the tested configurations and gives scale to those runs. The remote and local trials differ in duration, environment and route. With one short remote observation per setting, it would be misleading to claim a proven optimal batch size, an exact VPN penalty or a stable percentage improvement from the bars alone.

A stronger network benchmark would include warmup, repeated and interleaved trials, comparable run durations and workloads, latency distributions, CPU use, and separate network, rendering and decoding timings. To study training efficiency, it would also measure the actual GPU training workload and idle time.

## 7. How does resumable checkpoint transfer work?

A **checkpoint** is a saved training artifact. Depending on its schema, it can contain model parameters and other state needed to continue or evaluate training. The network prototype transfers a stable source checkpoint and validates the bytes before publishing the restored file.

The documented workflow is:

1. Compress the stable source using gzip.
2. Send the compressed bytes to the receiver.
3. If interrupted, retain the partial compressed file.
4. Resume sending at the receiver's saved **compressed-byte offset**.
5. Verify the compressed data and the decompressed checkpoint using SHA-256.
6. Publish the completed checkpoint only after the checks pass.

In the documented cross-laptop trial, the sender intentionally disconnected after **50,000 compressed bytes**. The next transfer resumed at offset 50,000 and the recovered file's SHA-256 matched the source. One recorded file was 4,379,345 bytes uncompressed and 3,983,694 bytes compressed.

This demonstrates recovery from that controlled interruption. It does not prove recovery from all failures, power-loss durability, delta checkpoint transfer, or automatic restart of training.

### What does SHA-256 establish?

SHA-256 produces a fixed-size digest from a file. Matching a digest against a separately trusted expected digest provides a practical integrity check for accidental corruption or a mismatched file. A checksum by itself does **not** authenticate who sent the bytes: an attacker who can alter both the data and an unauthenticated digest could replace both. SSH authentication and trusted source handling address different parts of the problem.

## 8. What is task-parallel inference?

**Inference** means using an already trained model to generate or evaluate outputs. In this prototype, a coordinator gives independent jobs—such as a specified requested color and seed—to workers that load a checkpoint and generate images.

The coordinator maintains a job plan, records worker and source fingerprints, verifies returned payload information, retries another endpoint after an error, and can resume a prior plan while reusing verified completed outputs. Each worker can produce a complete image for its assigned job.

This is **task parallelism**: different workers do different jobs. It is not model parallelism within one image's diffusion trajectory, and it is not synchronized data-parallel training with gradient averaging.

The documented two-worker functional check used two GPU laptops. Each worker generated one image from the same verified checkpoint; the two output hashes matched a prior single-worker run for that particular check. This is useful evidence that the plumbing worked for those jobs. It does not establish general cross-device bitwise reproducibility or a speedup: the single- and two-worker runs were not a controlled performance comparison.

The coordinator is ordinary deterministic software orchestration; it should not be presented as an autonomous AI agent unless such an agent was actually part of the measured workflow. Retry after an error also does not guarantee exactly-once execution: a lost response can result in a job being computed again.

## 9. What makes the networking contribution useful?

The engineering value is the **integration** of a reproducible synthetic-scene workflow with network transport, integrity checks, recoverable transfer and multi-worker job management. For a constrained research team with separate machines, this provides practical building blocks for moving artifacts and distributing independent work.

The research outcome remains the exposure question. Networking supports experiments; it does not by itself demonstrate that joint exposure affects generation. Neither the networking machinery nor the graph establishes a novel congestion-control algorithm or an end-to-end training speedup.

## 10. Likely evaluator questions

**Why TCP instead of just copying files manually?** A stream gives programs a structured producer–consumer interface and permits per-batch telemetry. Manual file transfer remains useful in other situations; the experiment does not establish that TCP streaming is always faster.

**Why is framing necessary if TCP is reliable?** TCP orders bytes but does not preserve application message boundaries. Framing tells the receiver where a logical message or payload begins and ends.

**Why both SSH and SHA-256?** SSH provides an authenticated, encrypted transport when configured and used correctly. A digest verifies data against a trusted expected hash; it does not by itself authenticate the sender.

**Does 7.81 Mb/s mean the VPN's maximum bandwidth?** No. It is the recorded PNG payload rate for one short scene-stream trial, excluding JSON and protocol overhead and including application work in elapsed time.

**Did both GPUs train one model together?** The documented two-worker check generated independent images. It was task-parallel inference, not synchronized training.

**Does the checkpoint resume automatically after the laptop restarts?** The described transfer protocol can resume a partial transfer when invoked again. The documented controlled test is not a guarantee of automatic training restart or recovery from every sleep or power-loss event.

## What to tell the audience

> “We built a networked research workflow with three parts: streaming synthetic scenes and annotations over framed TCP through an SSH tunnel, transferring compressed checkpoints with byte-offset resume and SHA-256 verification, and coordinating independent inference jobs across GPU workers. The scene-stream graph reports application throughput for local and cross-laptop tests; the measured rates include rendering and encoding. In a controlled checkpoint test, transfer resumed after a 50,000-byte interruption. Two GPU workers also completed a small functional inference check.”
