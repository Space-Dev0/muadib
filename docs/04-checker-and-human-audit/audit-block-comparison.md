# Exploratory paired checker coverage comparison

The 300-image pilot reused 75 generation seeds across four conditioning rules.
`exposure.audit_block_comparison` accepts one row per seed and condition, with
one human label and binary-or-null outputs from two fixed checkers. It rejects
duplicate or incomplete seed blocks, counts accepted outputs and accepted
disagreements, and compares coverage using a paired bootstrap that resamples
whole seed blocks. It does not combine multiple reviews or adjudicate labels.

The pilot's local input was constructed from the previously verified frozen
standard predictions, the post-hoc experimental rule-aware predictions,
original unanimous or single-review answers, the six selected follow-up
decisions, and the later `pilot-0279` follow-up. Original reviews remain
unchanged. Input and output are ignored in
`outputs/audit-pilot-results-800000/`; reproduce the summary with:

```sh
uv run python -m exposure.audit_block_comparison \
  --input outputs/audit-pilot-results-800000/seed-block-input-development.json \
  --output outputs/audit-pilot-results-800000/seed-block-comparison-development.json
```

The standard checker answered 95/300, the experimental checker 166/300. The
candidate answered all 95 standard-answered cases plus 71 additional cases;
their outputs agreed wherever both answered. After the post-selection follow-up
decisions, neither had an accepted disagreement against the assembled labels.
The candidate's coverage difference was 23.7 percentage points. A 2,000-draw
paired seed-block percentile bootstrap (seed 81217) gave 16.3–31.3 percentage
points for that difference, **conditional on treating the 75 seed blocks as
exchangeable**.

These labels and outputs were used to develop the candidate, and seven answers
were revisited after outcomes were examined. The bootstrap does not account for
reviewer error, adaptive tuning, other dependence, or sample-selection effects.
It is not an interval for checker error, independent validation, or a Stage 0
gate. Abstentions remain unknown outcomes and may differ systematically from
accepted images. A fresh, frozen, prospectively sampled human-labeled set with
broader benchmark coverage remains necessary.
