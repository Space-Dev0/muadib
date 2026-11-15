# Fresh 760k single-cell review

This is a prespecified, narrow follow-up to the eight post-hoc readiness
images. Generate red and blue conditions for each seed 81530 through 81545,
using the frozen 760,000-update checkpoint, EMA, guidance 1 and the existing
1,000-step DDPM sampler. The 32 outputs share 16 initial noise seeds across
colors; they are not 32 independent training runs. The seed and condition set
was fixed before the run and the pack includes every generated output, with no
selection based on visual quality or checker decisions.

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
