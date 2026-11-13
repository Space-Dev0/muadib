# Exploratory LP endpoint designs

`exposure.design_extremes` creates a diagnostics-compatible `designs` file:
the reference plus a minimum- and maximum-joint-exposure LP endpoint for each
rule. Every endpoint fixes the estimated singleton events at their reference
mixture values and respects a requested per-condition weight floor. Because
the singleton marginals are fixed on this matrix, endpoints also span that
rule's estimated feasible lift interval where the log lift is defined. The
objective differs for each rule: these are not jointly optimized, D-optimal,
representative, or necessarily distinct designs. Solvers can choose different
vertices when optima are non-unique. Rows share rules and designs; pooled
statistics depend on the chosen endpoint multiplicities.

```sh
uv run python -m exposure.design_extremes \
  --report outputs/exposure-57-pilot.json --reference outputs/uniform-57.json \
  --min-weight 0.001 --output outputs/design-extremes-57.json
uv run python -m exposure.design_holdout \
  --training outputs/exposure-57-pilot.json --holdout outputs/exposure-57-holdout-76001.json \
  --reference outputs/uniform-57.json --designs outputs/design-extremes-57.json \
  --output outputs/design-extremes-holdout-57.json
uv run python -m exposure.design_diagnostics \
  --report outputs/exposure-57-pilot.json --designs outputs/design-extremes-57.json \
  --output outputs/design-extremes-diagnostics-57.json
```

For this pilot's 57 rules, 115 rows of designs include 15 duplicate weight
vectors after rounding weights to nine decimals. In 6,555 rule-by-design
rows, median rule lift range was 1.10076, versus 0.09553 in the previous
20-design sample; 214 rows had positive lift, versus zero previously. The
maximum condition weight was 0.3785. These comparisons describe different,
dependent sets with unequal multiplicities and do not establish precision,
efficient selection or a stage result.

Training-matrix pooled correlation was −0.3232 with lift VIF 2.2212; a second
100-scene-per-condition measurement gave −0.3435 and 2.2107. Both exceed the
proposal's thresholds (`|r| ≤ 0.3`, `VIF < 2`) for these exploratory pooled
features, and breadth remains omitted. Maximum absolute singleton deviation
from the reference was 1.3e−13 on the training matrix but 0.07570 on the
independent-seed matrix. The latter highlights sensitivity to matrix estimates;
it is not a population error bound. The sampled distributions are extreme,
many endpoints duplicate others, their rows share designs and rules, and
matrix uncertainty is not propagated. `gate_status` in the holdout report is
`not_assessed`; this does not pass Stage 0.

An additional independently seeded matrix with 500 accepted scenes per
condition (`outputs/exposure-57-holdout-81205.json`) gave endpoint pooled
correlation −0.33034, lift VIF 2.28962, and maximum singleton deviation
0.06789. The earlier 20 sampled designs on the same holdout had correlation
−0.44545, lift VIF 11.58996, and maximum deviation 0.01082. These are
different dependent design collections, not a causal or uncertainty-adjusted
comparison. Both still miss the stated correlation/VIF thresholds in these
exploratory pooled calculations; breadth is omitted. Reproduce with
`exposure.measure` at seed 81205 and 500 samples per condition, then the
`design_holdout` command above with the desired design file. Reports are in
ignored `outputs/design-extremes-holdout-81205.json` and
`outputs/design-samples-holdout-81205.json`.

As a second direction check, fitting the same endpoint construction on the
500-scene seed-81205 matrix and evaluating on the separate seed-76001 matrix
gave training correlation −0.33879 and lift VIF 2.32921; the holdout gave
−0.34675 and 2.27176, with maximum singleton deviation 0.05504. The holdout
has only 100 scenes per condition. This is again an exploratory reversal of
matrix roles, not a registered gate evaluation or evidence that the candidate
templates are suitable. Outputs are `outputs/design-extremes-81205.json` and
`outputs/design-extremes-81205-holdout-76001.json`.
