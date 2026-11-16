# Clean spatial counterfactual diagnostic

The fresh generated-image pilot contained no negative relation-singleton
answers. This deterministic synthetic diagnostic directly creates negative
and near-tie cases with known object centers, then compares checker outputs to
`satisfies` on the original annotations. It does not use generated images or
human judgments and cannot pass the Stage 0 checker gate.

For both `left_of` and `above`, three separated red, blue and green objects
share one coordinate, while the blue object shifts by −5, −2, −1, 0, +1, +2
or +5 pixels along the tested axis. Every object takes each of the three
renderer shapes, for 27 shape combinations. This gives 378 scenes, with 54
negative relation-singleton ties (27 per axis) and 324 positive singleton
cases. Each scene is evaluated both for the unbound singleton and for a
red-to-blue bound relation; for the latter, the truth is negative for shifts
≤ 0 and positive for shifts > 0. Each design is a paired counterfactual,
not an independent random draw from the experimental distribution.

On these clean renders, the standard and variable-count experimental checkers
both answered all 756 scene-by-rule questions, including all 54 negative
singleton questions and all 108 bound ±1-pixel questions, with **zero accepted
disagreements** against renderer annotations. The result does not validate
shape or center detection on model-generated objects, occlusion, out-of-palette
colors or human perception, and it does not prove the assumed uncertainty
intervals hold there. It only closes this particular clean-render unit-test gap.

Reproduce from the repository root:

```sh
uv run python -m unittest tests.test_spatial_counterfactual_audit -q
uv run python -m exposure.spatial_counterfactual_audit \
  --output outputs/spatial-counterfactual-audit-new.json
```

The command refuses to overwrite an existing report. The local report is in
`outputs/spatial-counterfactual-audit.json`; outputs are ignored by Git.
