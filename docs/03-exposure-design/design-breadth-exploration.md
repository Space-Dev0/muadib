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

That fresh measurement completed and reproduced its own singleton and pair
counts. Applying the saved candidate without refitting produced marginal/lift
correlation 0.195 and VIFs 1.239 / 1.685 / 1.616 / 1.564 / 1.257 in the
feature order above. The full result and input hashes are in ignored
`outputs/design-breadth-fresh-82501.json`. This supports numerical stability
for that draw, but 300 scenes per condition remains far below the proposal's
approximately 100,000, and the concentrated target population, role-sensitive
breadth, grammar, dependence and uncertainty issues remain unresolved.
The same fresh draw shows maximum singleton-exposure deviations of 0.0654
for the uniform-reference group and 0.0484 for the fitted-reference group,
each measured against its own reference on that draw; the median per-design
maxima are 0.0239 and 0.0263. Exact equality on the original estimated matrix
therefore does not carry over exactly to this independent estimate. The
extended result including these deviations is in ignored
`outputs/design-breadth-fresh-82501-v2.json`.

Three further draws (seeds 82502–82504, 100 scenes per condition) were also
evaluated without refitting. Their largest VIFs are 1.735, 1.727 and 1.650,
and marginal/lift correlations are 0.181, 0.195 and 0.192. Maximum absolute
singleton deviations from each group's own reference range from 0.0747 to
0.0927 for uniform and 0.0679 to 0.0751 for fitted. These three small draws
are sensitivity checks, not confidence intervals; they reuse the same
renderer and template set. Ignored results are
`outputs/design-breadth-fresh-82502-v2.json` through `82504-v2.json`.

One draw uncovered a floating-point edge case: a mathematically valid mixture
of certain events summed to `1.0000000000000002` and was rejected downstream.
The mixture calculation now clips only tiny roundoff at the probability
boundary after validating inputs; values outside a `1e-12` tolerance still
raise an error. A regression test exercises certain rule, singleton and pair
exposures. The failed draw completed after the fix.

A larger new draw (seed 82505, 1,000 scenes per condition) gave maximum VIF
1.684 and marginal/lift correlation 0.180 with the same frozen weights. The
maximum singleton deviations from each group's own reference were 0.0656
(uniform) and 0.0615 (fitted), with median per-design maxima 0.0241 and
0.0307. These deviations are estimated on the new matrix; they do not by
themselves identify the exact population deviations or their uncertainty.
The result is in ignored `outputs/design-breadth-fresh-82505.json`.

To investigate that deviation, the endpoint designs and role-sensitive
allocation were refitted using the 1,000-scene matrix (seed 82505). With the
wider 0.05–4.0 rule-weight bounds, the refit has an effective 16.75 rules.
On the previously inspected 300-scene seed-82501 matrix it gives marginal/lift
correlation 0.162, maximum VIF 1.748, and maximum singleton deviations 0.0329
and 0.0327 for the uniform and fitted groups. This is a post-hoc sensitivity
comparison, not a fresh blinded validation: the seed-82501 results were seen
before this refit. Both weight bounds are reached, the proposal's approximate
100,000 scenes per condition has not been reached, and no approved choice of
target population or breadth definition is implied. The frozen refit and its
comparison are stored under ignored `outputs/design-breadth-refit-82505*.json`.

An additional local-optimizer sensitivity sweep on that same 1,000-scene
matrix tried rule-weight bounds of 0.15–2.5, 0.10–3.0 and 0.075–3.5 times
equal weight. On the already inspected 300-scene matrix, the first candidate's
largest VIF was 2.121. The second had largest VIF 1.879 and correlation 0.290,
with effective rule count 20.94; its correlation is close to the 0.3 limit.
The third had largest VIF 1.752 and correlation 0.224, with effective rule
count 17.78. These are additional post-hoc, local-optimizer results, not proof
of a minimal weight concentration, a fresh validation, or an approved change
to the target population. Reproduce with the refit command and `--frontier`.

As a definition sensitivity check, `--mode scene` instead collapses variable
names into nine distinct singleton scene events and counts a partner whenever
both singleton events hold independently on the same scene. This is invariant
to variable names, but differs from a shared-object conjunction: a red square
and a blue circle jointly satisfy the red and circle singleton events without
forming a red circle. Source singleton counts are verified; within-rule pair
counts need not match because the meaning differs. With the same frozen
weights on seed 82501, five VIFs are 1.291 / 1.698 / 1.669 / 1.386 / 1.111
and marginal/lift correlation is 0.195. This does not decide which breadth
definition is scientifically appropriate. The ignored sensitivity result is
`outputs/design-scene-breadth-fresh-82520.json`.

Reproduce from the repository root with the existing ignored reports:

```sh
uv run python -m exposure.design_breadth --report outputs/exposure-57-pilot.json --output outputs/breadth-pilot-new.json
uv run python -m exposure.design_breadth --report outputs/exposure-57-holdout-81205.json --output outputs/breadth-holdout-new.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run python scripts/explore-breadth-design.py --output outputs/design-breadth-exploration-new.json
uv run python -m exposure.measure --rules outputs/templates-57.json --samples 300 --seed 82501 --include-subsets --output outputs/exposure-fresh-new.json
uv run python -m exposure.design_breadth --report outputs/exposure-fresh-new.json --output outputs/breadth-fresh-new.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run python scripts/check-frozen-breadth-design.py --frozen outputs/design-breadth-exploration-82500.json --report outputs/exposure-fresh-new.json --breadth outputs/breadth-fresh-new.json --output outputs/design-breadth-fresh-new.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 uv run python scripts/explore-breadth-design.py --training-report outputs/exposure-57-fresh-82505.json --holdout-report outputs/exposure-57-fresh-82501.json --training-breadth outputs/breadth-57-fresh-82505.json --holdout-breadth outputs/breadth-57-fresh-82501.json --frontier
```

The exploration script reads the original generated filenames
`outputs/breadth-57-pilot-82500.json` and
`outputs/breadth-57-holdout-82500.json`, checks their source SHA-256 values,
and will not use files regenerated under the example names without explicitly
selecting or renaming them. Source reports and outputs are ignored artifacts.
