# Course review: implementation and evidence status

The suggested course review is a work plan, not evidence that proposed performance targets have been achieved. In particular, a claim that any checker reduces blur attrition to single digits requires an independently evaluated implementation and actual data. An alternative rule set does not itself fix the renderer's near-tautological relation singletons. A D-optimal objective cannot, by itself, guarantee identifiable effects, two-sided lift, or valid rule coverage. Stage 0 has not passed.

## Existing evidence and model comparison

The recorded 500,000-update matched evaluation compares all four cells at the same training budget, on the same 64 prespecified model samples, with labels, unresolved cases, and checker abstentions reported in [the evaluation](../06-mosaic/mosaic-matched-eval-82000.md). Comparing 500,000-update high cells against 150,000-update low cells would confound condition with training budget and would not isolate complexity. It may be a historical descriptive comparison, but is not a matched experiment. One training-seed block and fixed data prevent generalization to model or dataset populations.

The GPU currently has a previously authorized independent training task running. No new long diffusion run was launched in response to this review. A claimed 20-minute overfit win needs an actual controlled probe and assessment; capacity, clipping, and positional encoding cannot be called the cause without separating them.

## Instruments and review provenance

The restricted 800k audit had 205/300 standard-checker abstentions; the pre-existing blur-1.0 stress test was on renderer images, not on those same 300 generated images. These are different samples and denominators. Four 760k follow-up CSVs and the four matched-evaluation review CSVs were byte-identical within their respective packs, so hashes alone do not demonstrate four independent reviewers or permit an agreement estimate. Preserve the source data and any unresolved human labels. See [the pilot audit](../04-checker-and-human-audit/pilot-audit-800000.md) and the matched evaluation above.

Both Gamma definitions have zero gap under their stated product null wherever the transformations are defined. The original arithmetic criticism is retracted; see [composition gap](../01-instruments/composition-gap.md). A product null may still be inappropriate for predicates with shared bindings.

Private host addresses were removed from the current `docs/` working tree. Git history and other locations were not rewritten; anyone requiring removal from history must assess and separately plan that operation.

## Work performed

- Networking now has a CPU-rendered TCP streaming prototype, a request-driven conditional renderer adapter, and gzip checkpoint transfer with SHA-256 and interrupted-transfer resume. An experimental opt-in wrapper connects the conditional adapter to the pilot trainer without changing its source, and has passed tiny same-host CPU parity and resume checks; the protocols are for trusted peers or authenticated tunnels. A longer localhost run processed 900,000 scenes in 843 seconds and measured approximately 1,082, 1,061 and 1,060 scenes/s for batch sizes 2, 8 and 16, respectively, including rendering and encoding. A separate cross-laptop stream through an SSH tunnel measured 582, 801 and 926 scenes/s, and 4.91, 6.76 and 7.81 Mb/s of PNG payload, respectively, for 300 scenes each. These short measurements include rendering and encoding; they are not physical link capacity or integrated training performance. GPU idle latency remains unmeasured. See [networking details](../../ops/README.md).
- [Image diagnostics](../01-instruments/image-diagnostics.md) implement radial FFT power and regional errors for explicitly aligned pairs. Existing randomly generated images and unrelated clean renders are not aligned reconstruction pairs. A synthetic blur demonstration measures an instrument, not a model or checker accuracy improvement. A separate bounded CUDA foreground segmentation run on 100 held-out synthetic scenes achieved pixel IoU 0.9427 clean and 0.9347 at blur 1; foreground-mask IoU alone does not measure object or rule accuracy; the rule-adapter evaluation is reported separately below.
- [Factorial templates and D-optimal exploration](../03-exposure-design/factorial-d-optimal-exploration.md) provide 288 endpoint-bound templates with a 216/72 template split. The audit remains blocked by relation saturation. D-optimal pilot VIFs remain above 2 and its effective rule count is 16.07; this does not establish a valid identified design.

- A frozen foreground-segmentation-to-rule adapter was compared with the existing
  checker on a new set of 200 synthetic scenes and on the previously human-reviewed
  300 generated images. On blur-1 synthetic scenes, it abstained 147/200 versus
  172/200 for the baseline on the same scenes, but made 17 false-positive and 37
  false-negative rule decisions among 477 accepted correlated rule decisions;
  the baseline had zero accepted decision errors in this sample. On the reviewed
  generated images the adapter answered 54/300, versus the baseline's 95/300.
  Among 53 human-resolved accepted adapter decisions there were 42 TP, 10 TN,
  1 FP and 0 FN; one accepted image had unresolved human disagreement. These are
  exploratory, selected-sample results, not an independent validation or evidence
  that the checker gate passed.
