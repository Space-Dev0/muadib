# Exploratory bounded spatial relations

The existing benchmark's unbound existential `left_of` predicate is nearly
always true under its three-object renderer (analytic probability 0.999271).
This makes its singleton exposure a weak discriminator. The experimental
module `exposure.spatial_sector` explores a different event; it does **not**
change the legacy rule language or establish that the Stage 0 gate passes.

For a subject and target, the candidate requires the subject's displacement
to lie within 15 degrees of the requested cardinal direction and the distance
between their centers to be between 14 and 60 pixels, inclusive. The
existential form searches distinct objects; optional filters bind attributes
to each endpoint. Image coordinates increase to the right and downward.

We sampled 20,000 independent base-renderer scenes with seed 82810. The
intervals below are pointwise 95% Wilson intervals, conditional on the
renderer and chosen parameters; they are not simultaneous intervals or
evidence about generated images.

| Event | Count | Estimate | 95% interval |
| --- | ---: | ---: | ---: |
| First object left of second | 2,444 | 0.1222 | [0.1177, 0.1268] |
| Some distinct object left of another | 13,142 | 0.6571 | [0.6505, 0.6636] |
| Red circle left of blue square | 102 | 0.0051 | [0.0042, 0.0062] |

The existential comparison is like-for-like at the quantifier level: 0.6571
for this new sector versus 0.999271 for the old inequality. The events have
different geometry and cannot be substituted without revising the research
protocol. The much rarer bound event also warns that naive rejection sampling
could be expensive; its frequency must not be mistaken for evidence of a
balanced or feasible benchmark.

## Endpoint support preflight

The separate `exposure.sector_support` experiment counts, once per scene, each
existential event formed by two directions (`left`, `above`) and all 12 × 12
ordered color–shape endpoint combinations, including identical endpoint
attributes on distinct objects. It draws directly from the existing base
renderer, without conditioning on any rule. Thus it measures the probability
of a candidate rejection-sampling event, not performance of a trained model.

Two independently seeded 200,000-scene simulations gave these results:

| Seed | Events | Minimum count | Median count | Maximum count | Zero counts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 82811 | 288 | 910 | 990.5 | 1,070 | 0 |
| 82812 | 288 | 915 | — | 1,072 | 0 |

For seed 82811 the observed event probabilities range from 0.00455 to
0.00535. Their reciprocal suggests roughly 187–220 candidate scenes per
acceptance for simple independent rejection sampling, using point estimates;
actual runtime and tail costs require direct measurement. These extrema are
selected across 288 correlated events, so the per-event Wilson intervals in
the output are not simultaneous guarantees. No observed zeros supports, but
does not prove, universal support or balanced conditional datasets. The
second simulation checks sampling variability; it is not an independent
implementation or a confirmatory benchmark evaluation.
The ignored output files have SHA-256 hashes `476f67e96e4459f5ada34de3cbaa55fd439fceaa62c79ccc5302ac81e75ce958`
and `4e89f4d9c6d656474d4f9643e9e594364a7cd091ec250705da03dfa9227362f7`,
respectively.

```sh
uv run --no-sync python -m exposure.sector_support --samples 200000 \
  --seed 82811 --output outputs/sector-support-82811.json
uv run --no-sync python -m exposure.sector_support --samples 200000 \
  --seed 82812 --output outputs/sector-support-82812.json
```

Reproduce with:

```sh
uv run --no-sync python -m exposure.spatial_sector --samples 20000 --seed 82810 \
  --output outputs/spatial-sector-82810-v2.json
```

The command refuses to overwrite an existing output. Before this candidate
can support the Stage 0 gate, the team must define a versioned rule grammar,
counterbalance endpoint attributes and directions, establish conditional
sampling support and cost, rerun exposure design and held-out audits, and
validate the image checker on the resulting generated scenes. Current
four-mixture held-out violations and checker limitations remain unresolved.
