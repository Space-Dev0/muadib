# Exploratory factorial relational subset

`python -m exposure.factorial_relations --output outputs/factorial-relations-24.json`
constructs 24 candidate rules: every combination of four colors, three shapes,
and two relation kinds. Color belongs to the subject and shape to the distinct
object, so both relation endpoints have attributes. Every color–shape pair
appears once with each relation. This addresses the relation-kind and endpoint
binding confound in one complexity cell; it does not include same-object
color–shape binding, other complexity cells, a held-out split, texture, or a
third relation. It is not a final benchmark.

The relation singleton, evaluated on its own, still independently rebinds its
variables and remains near-tautological under the existing renderer. Binding
the endpoints in the full rule does not solve that singleton saturation.
Sampling support, exposure distributions, checker validity and design
identifiability must be measured before any research use. Example exploratory
measurement (500 scenes per condition; not proposal-scale):

```sh
uv run python -m exposure.measure --rules outputs/factorial-relations-24.json \
  --samples 500 --seed 81201 --include-subsets --output outputs/exposure-factorial-24-81201.json
```

The result is instrument-development evidence, not a decorrelation-gate or
Stage 0 result. Do not replace the final benchmark without addressing the
remaining cells and the saturated relation singleton semantics.

The specified 500-scene run completed for all 24 conditions; average sampling
attempts per accepted scene ranged from 4.568 to 5.584 across conditions.
Across its 576 condition-by-rule relation-singleton entries, estimated
probabilities ranged from 0.996 to 1.000 (mean 0.99983). These entries share
scenes and duplicate relation events, so they are not 576 independent
measurements; they confirm that this subset does not solve saturation.

An unconditioned baseline can be reproduced with:

```sh
uv run python -m exposure.factorial_baseline --samples 10000 --seed 81202 \
  --output outputs/factorial-baseline-81202.json
```

In this run the 24 full attributed rules ranged from 0.1863 to 0.2030
(mean 0.1952), while the independently rebound `left_of` and `above`
singletons were 0.9991 and 0.9996. These are distinct events: binding
attributes to relation endpoints in the full rule does not change the
unbound singleton event used in the current composition baseline. Every rule
is evaluated on the same scenes; reported Wilson intervals are pointwise,
not simultaneous, and rule estimates are correlated.
