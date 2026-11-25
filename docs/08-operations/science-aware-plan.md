# Balanced assignments and cross-device replay plan

`ops.science_plan` prepares a **static, pinned baseline and a separate audit**
for an existing frozen inference plan. It is a planning tool, not yet evidence
that the scheduler improves runtime or preserves a scientific conclusion.

```sh
uv run python -m ops.science_plan \
  --plan plan.json --workers workers.json --output outputs/science-design \
  --audit-per-stratum 2 --selection-seed 42
```

The input formats are the existing `ops.adaptive_inference` formats. Every
original logical job is assigned exactly once to a compatible worker; none is
added, deleted, or edited. The greedy allocator minimizes the current count
for the job's stratum, then the worker's total assigned count, with a stable
worker-ID tie break. It is a heuristic, not a global optimum, and checkpoint
compatibility may prevent perfect balance. Define strata to encode the research
conditions whose worker allocation matters; an overly broad label cannot be
balanced internally. Incompatible or case-colliding worker IDs fail validation.

Before any results are observed, hash-based ordering selects up to the requested
number of jobs in each stratum. Each selected job has:

- its original `base` run on its assigned worker;
- a separate `repeat` run on that same worker;
- a separate `cross` run on every other compatible worker.

All phases use the original job dictionary and seed. They are in distinct plans
and run directories, so replay images cannot silently inflate the original
scientific sample count. If a checkpoint is available on only one worker, the
manifest has an empty `comparison_workers` list; a same-worker repeat is not a
cross-device comparison. Hash selection is deterministic for the specified seed,
not automatically an unbiased statistical sample.

The output includes `manifest.json`, the full `base-plan.json`, a
`base-assignments.json` map, one-worker configurations in `workers/`, and only
nonempty coordinator-compatible plan shards under `base/`, `repeat/`, `cross/`.
Each `audit_pairs` entry links the original job ID, stratum, primary worker,
repeat worker and comparison workers. The source plan hash is SHA-256 of
UTF-8 `json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False)`;
it identifies that serialized plan, including the order of its job list.

With the corresponding SSH tunnel already established, run each phase in its
own output directory, for example:

```sh
uv run python -m ops.adaptive_inference \
  --plan outputs/science-design/base/w1-plan.json \
  --workers outputs/science-design/workers/w1.json \
  --output outputs/science-runs/base/w1
uv run python -m ops.adaptive_inference \
  --plan outputs/science-design/repeat/w1-plan.json \
  --workers outputs/science-design/workers/w1.json \
  --output outputs/science-runs/repeat/w1
```

Run the generated `cross/<worker>-plan.json` shards analogously under
`outputs/science-runs/cross/<worker>`. Do not run a nonexistent shard: workers
with no jobs in a phase do not receive an empty plan. A one-worker shard is
pinned; an unavailable worker leaves its jobs incomplete rather than silently
changing the intended assignment. This baseline has no automatic failover.

Same seed across different hardware or PyTorch versions does **not** ensure an
identical random-number stream or image. A differing SHA-256 indicates different
bytes, not a perceptible or scientifically meaningful difference. Evaluate
semantic disagreement only with a separately validated evaluator or independent
human labels, preserve missing outputs, and do not treat image replicas as
independent training runs. Avoid causal or novelty claims without appropriate
controls and comparison with prior work.
