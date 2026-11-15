# Fresh 760k single-cell review

This is a prespecified, narrow follow-up to the eight post-hoc readiness
images. Generate red and blue conditions for each seed 81530 through 81545,
using the frozen 760,000-update checkpoint, EMA, guidance 1 and the existing
1,000-step DDPM sampler. The 32 outputs share 16 initial noise seeds across
colors; they are not 32 independent training runs. The seed and condition set
was fixed before the run and the pack includes every generated output, with no
selection based on visual quality or checker decisions.

The run completed on the CUDA laptop and the full batch was copied to
`outputs/mosaic-guidance-fresh-760k-81530/`. The packager verified all 32 image
and pixel hashes and wrote reviewer ZIPs under
`outputs/mosaic-guidance-fresh-review-760k-81530/`. Experimental checker
predictions were frozen in the private directory before distributing the
pack; their file SHA-256 is
`47e12ff20b3fd20b212a4c2109d64e67b90ff402f72deea7d8bffb9bc2a2fa0e`.
Human review results are summarized below.

## Returned reviews

Four complete review files were returned, each with the same SHA-256
`07b151d381253bda02443669bd3dfe9ddf825302ab5483883cf0f348558ef87a`.
The user reported the reviews complete. Because the files and review archives
contain no identifying metadata, independent authorship cannot be verified from
the artifacts alone. The reviewers answered yes to both questions for 31 of
32 images, and no to both for `readiness-009`. There were no unsure responses
or recorded disagreements. For `readiness-007`, each file says "yes, objects
are touching" for target-circle presence; the scorer interprets the explicit
yes and preserves the original files and their hashes. Pairwise agreement is
32/32 for each question and each reviewer pair.

The frozen experimental checker answered true on 28 images, all rated yes for
target-circle presence. It abstained on four: three rated yes (`readiness-007`,
`-010`, `-032`) and the one rated no (`-009`). The 28 accepted answers contain
no disagreements with the reviews; no negative image was accepted, so the
sample cannot estimate a false-accept rate. Checker abstention reasons were
unrecognized foreground color on one, one connected component on one, and
unsupported size or occlusion on two. These labels do not validate object-count
predictions; the checker was not asked that question. The 32 images are paired
across 16 seeds and come from one model and one simple training cell. Neither
the 31/32 review result nor accepted-only checker agreement meets the Stage 0
gate or establishes quality for more complex scenes.

The frozen prediction hash above was checked again before comparison. Scored
votes and file hashes are in the ignored `score.json` next to the packs; the
original review files are unchanged.

The package contains four identical blinded image sets. Each reviewer answers
whether the requested-color circle is visible and whether exactly two distinct
objects are visible. Reviewer answers should be kept as separate files and
reviewers should work independently. The private directory and any checker
predictions must not be given to reviewers before they finish. Record unsure,
disagreements and abstentions; do not treat detector failures as negative human
answers or majority agreement as infallible ground truth.

Once all four files have been returned, score them with four distinct paths:

```sh
uv run --no-sync python -m exposure.mosaic_guidance_review_score \
  --pack outputs/mosaic-guidance-fresh-review-760k-81530 \
  --reviews reviewer1.csv reviewer2.csv reviewer3.csv reviewer4.csv \
  --output outputs/mosaic-guidance-fresh-review-760k-81530/score.json
```

The scorer checks the private images against the local provenance manifest,
checks complete unique sample IDs and response values, retains vote counts and
reports pairwise agreement excluding unsure answers. The manifest is not a
cryptographic attestation of reviewer identity, independent review, or image
generation. The scorer does not decide disputed cases or evaluate checker error.

```sh
UV_CACHE_DIR=/tmp/muadib-uv-cache uv run --no-sync python -m exposure.mosaic_guidance_review_pack \
  --source outputs/mosaic-guidance-fresh-760k-81530 \
  --output outputs/mosaic-guidance-fresh-review-760k-81530 --fresh-81530
```

The packager requires all 32 exact color/seed/guidance combinations, checks
image and pixel hashes against the report and checks the recorded checkpoint
hash. The report is a provenance record, not independent proof of how each
image was generated. This sample tests only a single model trained on one
low-complexity, two-object cell. It cannot establish the proposal's 300-image
complexity-stratified checker gate, a MOSAIC replication, or a Stage 0 pass.

Freeze experimental checker decisions before sharing packs or collecting new
reviews. Keep this output private until review is complete:

```sh
uv run --no-sync python -m exposure.mosaic_guidance_predictions \
  --pack outputs/mosaic-guidance-fresh-review-760k-81530 \
  --output outputs/mosaic-guidance-fresh-review-760k-81530/private/predictions.json
```

These decisions concern the requested-color circle only, and `null` means
abstention. The checker source hash is recorded, but dependencies and the
runtime are not independently attested. Compare with human labels only after
review, and retain disagreements and abstentions.
