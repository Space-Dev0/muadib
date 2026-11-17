# Resumable shared-assignment exposure measurement

The fast estimator evaluates the 57 rules and their subsets from shared
assignments of the existing three objects to variables `a`, `b` and `c`. Every
partial distinct binding extends to one of the six full assignments, so the
boolean result agrees with independently evaluating each rule. The tool refuses
other variable names and scene sizes. It uses the same conditioned-scene
sampler and RNG stream as the original estimator.

A 57-rule, 100-scenes-per-condition run at seed 43 reproduced the complete
original report exactly, including attempts, rule and subset counts and Wilson
intervals. A separate 25-scenes-per-condition local timing produced identical
reports and measured 2.52 seconds for the original and 0.30 seconds for the
shared-assignment implementation (about 8.4× on that machine and workload).
This short benchmark does not guarantee throughput for longer runs or other
machines. Tests also cover interrupted-state recovery and matching pair counts.

The optional state file is atomically updated after each conditioning column,
including the RNG state and completed counts. Resume checks the input rules,
run parameters, Python/NumPy versions, and relevant source fingerprints;
interruption can require repeating the unfinished column. Keep the state file
and choose new report paths rather than overwriting completed reports. A state
file isn't a substitute for independent validation or for keeping the source
code used to produce it.

A fresh seed-82511 measurement with 100,000 accepted scenes per condition is
planned against the previously frozen exploratory correlation-margin candidate
`outputs/design-breadth-margin-82540.json` (SHA-256
`fae16854df605094a0c9f19c42cabe1a75bb06a989eddf752cfba73ff6b91126`).
This creates a new renderer draw; it is not an external benchmark, a
confirmatory preregistration, approval of the reweighted rule population, or
a resolution of the grammar and breadth-definition issues. The report and
role-sensitive breadth counts are generated together from the same scenes.

The seed-82511 run completed all 57 conditions, for 5.7 million accepted
scenes in total and 57,827,027 rejection-sampling attempts. The completed report SHA-256 is
`aa1f933407b97acf42878f6a0b729b4ea03c0073694bdb72f19038c3e64d06b6`;
the breadth artifact SHA-256 is
`69223ee5fe3198db5883cb1e94c0bc57a4d8b0bea94d445ac7f48add90de13d6`.
Applying the frozen candidate without refitting gave correlation 0.237 and
five VIFs between 1.462 and 1.889. Maximum estimated singleton deviations
from the corresponding references were 0.0101 and 0.00749. Adding a shared
reference indicator increased the largest VIF to 4.887. The result is stored
in ignored `outputs/design-breadth-margin-fresh-82511.json`; these are
estimates for a concentrated, unregistered candidate, not a Stage 0 gate pass.

```sh
uv run python -m exposure.measure_fast --rules outputs/templates-57.json \
  --samples 100000 --seed 82511 --state outputs/fast-measure-82511.state.json \
  --output outputs/exposure-57-fresh-82511.json \
  --breadth-output outputs/breadth-57-fresh-82511.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run python scripts/check-frozen-breadth-design.py \
  --frozen outputs/design-breadth-margin-82540.json \
  --report outputs/exposure-57-fresh-82511.json \
  --breadth outputs/breadth-57-fresh-82511.json \
  --output outputs/design-breadth-margin-fresh-82511.json
```
