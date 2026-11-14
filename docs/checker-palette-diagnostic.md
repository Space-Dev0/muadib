# Palette and geometry abstention diagnostic

The MOSAIC preview images are too distorted for useful human review. This
exploratory diagnostic asks whether merely relaxing the checker palette guard
would make them usable; it does not change the standard checker. At tolerance
1000 the guard is effectively removed, but the remaining detector errors are
only the first failure encountered, not a complete diagnosis.

On all 16 unreviewed multiscale previews, the current tolerance 40 yields 16
palette rejections. At tolerance 1000, none is answered: 12 fail component count
and four fail size or occlusion. Thus removing the palette guard does not make
this detector applicable to those previews. These are not labels or accuracy
scores, and relaxed tolerances may introduce incorrect accepted predictions.

On the previously reviewed 40-image four-rule validation set, tolerance 40
answers 28 whole-rule questions and tolerance 45 answers 30, with no observed
accepted disagreements at either setting. Because these human labels have now
been used to explore this parameter, this is development evidence and cannot
validate a new threshold. Retain the existing checker until a frozen candidate
has been evaluated on fresh, representative human-reviewed images.

Reproduce with new output paths:

```sh
uv run python -m exposure.checker_palette_probe --pack outputs/mosaic-multiscale-review-packs-81322 --kind mosaic --output outputs/palette-mosaic-diagnostic.json
uv run python -m exposure.checker_palette_probe --pack outputs/audit-validation-packs-800000 --kind reviewed --output outputs/palette-reviewed-diagnostic.json
```
