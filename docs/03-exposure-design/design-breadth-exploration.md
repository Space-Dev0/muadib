# Exploratory role-sensitive breadth and five-covariate design

The proposal defines predicate breadth as `exp(H)` for the distribution of
co-occurring predicates, but does not uniquely specify a partner universe,
variable-role equivalence, how multiple partners per scene are counted, or how
predicate breadth becomes a rule covariate. This diagnostic makes those choices
explicit to test the existing candidate; it is not a registered definition or
Stage 0 gate result.

The partner universe contains the 23 distinct *syntactic* atomic predicates
in the current 57-rule table. For each predicate `p`, each other predicate `q`
receives mass proportional to `n({p,q})` under the design mixture. A scene may
contribute to multiple partners. Shared variable names bind to the same object;
distinct names bind to distinct objects. Variable-role duplicates remain
separate partner events. The rule covariate is the mean of its atoms' log
breadths, equivalent to the log geometric mean. These choices can change the
answer, and the role vocabulary is not invariant to arbitrary variable
renaming. They should not silently become the confirmatory estimand.

`exposure.design_breadth` regenerates the exact same conditioned scenes as each
source exposure report. It computes each pair via all distinct assignments and
checks every originally recorded singleton and within-rule pair count against
the report. Both the 100-scene-per-condition fitting report (seed 43) and
500-scene-per-condition holdout report (seed 81205) reproduced all of those
counts. Tests separately compare the pair calculation with the rule evaluator.
This is a second Monte Carlo measurement, not an external benchmark; the
underlying 57-template grammar and renderer have documented validity problems.

For the previously fitted two-reference rule allocation, adding this breadth
covariate gives the following weighted pooled VIFs:

| Matrix | log marginal | log lift | attributes | relations | log breadth | marginal/lift r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fitting | 2.020 | 1.849 | 1.522 | 2.452 | 2.653 | 0.173 |
| Holdout | 1.965 | 1.908 | 1.573 | 2.567 | 2.669 | 0.161 |

A further bounded optimization, fitting all five VIFs on the fitting matrix with
rule weights restricted to 0.2–2.0 times equal allocation, yields fitting VIFs
2.027 / 1.949 / 1.639 / 2.259 / 2.073, and holdout VIFs 2.029 / 2.020 /
1.684 / 2.353 / 2.083, respectively; the marginal/lift correlations are 0.222
and 0.210. It also changes the rule-weighted target population (effective rule
count approximately 31.9 of 57). This optimizer is local and does not prove that
no feasible allocation exists. Its candidate does not meet the proposal's
VIF < 2 criterion for all five features. The holdout has been inspected and is
not a fresh confirmatory test of further tuning. Neither candidate resolves
reference-group adjustment, uncertainty, dependence or the target estimand.

With wider exploratory bounds of 0.05–4.0 times equal rule weight, a local
refit reaches VIFs 1.201 / 1.621 / 1.575 / 1.529 / 1.233 on the fitting
matrix and 1.210 / 1.678 / 1.616 / 1.561 / 1.249 on the holdout, in the
same feature order; marginal/lift correlations are 0.177 and 0.181. This is a
promising numerical candidate, but the effective rule count falls to about
15.3 of 57 and both weight bounds are reached. It substantially changes the
rule-weighted population, was chosen after inspecting the earlier holdout, and
uses the unregistered breadth definition above. The original uncertainty,
reference-adjustment, grammar and dependence limitations remain. It does not
pass the Stage 0 gate or justify changing the target population by itself.
The frozen candidate weights and designs are in ignored
`outputs/design-breadth-exploration-82500.json` with SHA-256
`b170c937460ae03a9a4073ce035334c0289ebc951d4cef3ce32306b83ec1e729`.
After saving this candidate, a fresh seed-82501, 300-scenes-per-condition
measurement was started to check sensitivity without refitting the weights.
This is an independent Monte Carlo draw from the same renderer and rules, not
an independent benchmark or a confirmatory hypothesis test.

Reproduce from the repository root with the existing ignored reports:

```sh
uv run python -m exposure.design_breadth --report outputs/exposure-57-pilot.json --output outputs/breadth-pilot-new.json
uv run python -m exposure.design_breadth --report outputs/exposure-57-holdout-81205.json --output outputs/breadth-holdout-new.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run python scripts/explore-breadth-design.py --output outputs/design-breadth-exploration-new.json
```

The exploration script reads the original generated filenames
`outputs/breadth-57-pilot-82500.json` and
`outputs/breadth-57-holdout-82500.json`, checks their source SHA-256 values,
and will not use files regenerated under the example names without explicitly
selecting or renaming them. Source reports and outputs are ignored artifacts.
