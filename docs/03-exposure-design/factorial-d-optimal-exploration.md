# Factorial rules and D-optimal allocation: exploratory only

`exposure.templates_v2` generates 288 rules crossing all 12 color/shape
combinations on each of two distinct object variables with `left_of` and
`above`. Both endpoints carry their own color and shape predicates. It groups
both relation variants of each ordered pair into the same partition. A
preassigned modular split yields 216 training and 72 held-out templates;
every color/shape pair occurs in both partitions and relation kinds are
balanced within each. A template split is **not** evidence that generated
scenes, image datasets or model-training runs were separated.

The existing `benchmark_audit` now accepts `--split-manifest` and verifies
nonempty disjoint exhaustive name lists and checks for canonical semantic
rule overlap. This validates metadata, not actual split use or scene leakage.
Without a manifest the old result is unchanged. The audited factorial set
covers 12/12 co-bound pairs and does not have the earlier relation-kind versus
binding-class confound. The unchanged renderer still has an analytic relation
singleton exposure of 0.999271; 10,000 simulated base scenes with seed 82651
estimated 0.9995 for each kind. The saturation finding remains blocking and
`confirmatory_ready` remains false. This set has 4 attribute predicates plus
one relation throughout: it is not a balanced complexity benchmark. A small seed-82652 sampling smoke check drew 10 accepted scenes per rule: all 2,880 draws succeeded within 10,000 attempts, with a median 36 and maximum 637 attempts. This small check does not establish tail bounds or benchmark validity.

`exposure.design_optimal` implements an exploratory D-optimal allocation over
provided fixed feature rows, with simplex, minimum weights, and optional linear
bounds. It rejects rank-deficient inputs and validates solver output; maximizing
log determinant cannot make a genuinely collinear model identifiable. It does
not itself generate mixture endpoints, vary exposure within rule, handle
binomial observation weights, or validate a causal estimand.

Using the existing seed-82511 100k-scene matrix and the old `uniform_reference`,
we compared D-optimal between-rule allocation to equal weighting of 57 rules.
The fixed candidate matrix had an intercept plus standardized log mean marginal,
log lift, attribute count, and relation count; weights were restricted between
0.05/57 and 4/57. The optimized log determinant increased from -4.5710 to
-2.6438 (standardization matters to this absolute scale). Effective rule count
fell to 16.07. The four VIFs were [3.62, 17.72, 5.93, 15.81], compared with
[5.34, 20.28, 7.39, 21.33] under equal weights. All exceed the proposed VIF < 2
gate, so this result does not support an identified, decorrelated design.
It is an in-sample exploration over rules at one fixed reference, not a
within-rule intervention or a held-out result.

Reproduce using the existing ignored artifacts (choose fresh output names):

```sh
uv run python -m exposure.templates_v2 --rules-output outputs/new-v2-rules.json --manifest-output outputs/new-v2-manifest.json
uv run python -m exposure.benchmark_audit --rules outputs/new-v2-rules.json --split-manifest outputs/new-v2-manifest.json --samples 10000 --seed 82651 --output outputs/new-v2-audit.json
PYTHONPATH=. uv run python scripts/check_factorial_support.py --rules outputs/new-v2-rules.json --output outputs/new-v2-support.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run python scripts/evaluate_d_optimal_pilot.py --report outputs/exposure-57-fresh-82511.json --designs outputs/design-breadth-multimatrix-100k-82570.json --output outputs/new-d-optimal.json
```

The D-optimal input artifact hashes are recorded in the generated JSON. These
reports and endpoints are ignored local experiment outputs; missing files must
be regenerated or recovered before the command can run elsewhere.
