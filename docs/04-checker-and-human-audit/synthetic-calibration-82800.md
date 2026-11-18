# Balanced synthetic checker calibration candidate

`exposure.checker_calibration` builds a metadata-labeled 500-image synthetic set for **same-object color–shape conjunctions**. It is not a human-labeled set, does not include model-generated images or relational rules, and does not validate the checker for the research gate.

The fixed seed-82800 set has 300 train, 100 validation and 100 test images. Each split is balanced positive/negative, and contains equal numbers from four scenarios: a target with unrelated distractors, a correct target with competing same-color/same-shape distractors, a wrong-binding negative with the color and shape on different objects, and a missing-shape negative. Each scenario is crossed with clean, blur 0.5, blur 1.0, brightness 1.1 and Gaussian noise (standard deviation 10 in 8-bit channels). The target combinations and positions are seeded and randomized. The generator checks metadata truth with rule semantics and rejects identical **clean** raw-pixel images across all splits. This does not exclude near-duplicates or establish independence from other datasets. Since perturbations can change visual recognizability, metadata truth is not a substitute for independent human judgments.

The existing standard checker and experimental variable-count checker were run unchanged on the 100-image test split. Each truth category contains 50 images. All six entries are shown, including zeros:

| Checker | Truth | Yes | No | Abstain |
| --- | --- | ---: | ---: | ---: |
| Standard | Positive | 25 | 0 | 25 |
| Standard | Negative | 0 | 24 | 26 |
| Experimental | Positive | 33 | 0 | 17 |
| Experimental | Negative | 0 | 30 | 20 |

The test set contains 20 images per perturbation. On clean images, both checkers answered all 20 consistently with metadata. At blur 1.0 the standard checker answered 3/20, while the experimental checker abstained on 20/20. No wrong accepted decisions occurred in this small synthetic sample; that is not a population error guarantee. Coverage and correctness are conditional on the renderer, target strata, and image perturbations. Absence of false accepts among accepted cases does not characterize abstentions or establish performance on generated images. The two checkers' accepted subsets can differ. These hard-decision checkers do not expose a calibrated continuous score, so an ROC curve is not available from these outputs.

The generated manifest SHA-256 is `ce10c4451d328f56600dd8dbf0ae499d744356a7deb18f767bb71c36d1046dc8`. The complete scored test report SHA-256 is `92022b62abcaedaa5f3d0acb9f2499af2f0bfe8021d92d3b9b23e4134fb9e8bb`. The report records the checker source hash, evaluation source hash, all individual decisions, strata and labels. Dataset and reports are ignored `outputs/` artifacts. Both existing checkers were run without fitting on this set; this test is still exploratory after earlier development on similar renderer scenes, not a fresh independent confirmation.

Reproduce using new paths:

```sh
uv run --no-sync python -m exposure.checker_calibration generate \
  --seed 82800 --output outputs/checker-calibration-new
# Use the manifest SHA-256 printed by the command below.
uv run --no-sync python -m exposure.checker_calibration evaluate \
  --dataset outputs/checker-calibration-new --expected-sha256 MANIFEST_SHA256 \
  --split test --output outputs/checker-calibration-new-test.json
```

Generation and result writing refuse existing paths; interrupted generation can leave a partial directory. Tests cover deterministic data, positive/negative split balance, unique clean-scene hashes, all six outcome cells, and tampered manifest/image rejection. Before any gate decision, use an independently sampled generated-image test set with separately attributable blinded human reviews, include hard negatives and relational predicates, specify minimum per-class coverage/error targets and uncertainty, and preserve abstentions.
