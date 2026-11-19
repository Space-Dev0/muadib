# Paired synthetic overlap stress test

The earlier balanced synthetic calibration used grid-separated objects. This
additional exploratory test asks how the same checker behaves when two
objects overlap. It does not establish the Stage 0 checker gate.

`exposure.checker_overlap` constructs 200 metadata-labeled pairs, balanced 100
positive / 100 negative for a requested same-object color–shape conjunction.
Positive scenes contain a target and competing attributes; negatives place
the requested color and shape on different objects. Ordered object attribute
sequences are unique across pairs. For each pair, all labels and drawing order
are fixed: in the control the first two centers are 20 pixels apart, while in
the treatment they are four pixels apart; the third object stays in place.
Shapes have nominal width 12 pixels. Other geometry, perturbations, overlap
fractions, object counts and predicate kinds are not covered.

| Checker | Scene | Positive yes / no / abstain | Negative yes / no / abstain |
| --- | --- | --- | --- |
| Standard | Separated | 100 / 0 / 0 | 0 / 100 / 0 |
| Standard | Overlap | 0 / 0 / 100 | 0 / 0 / 100 |
| Experimental | Separated | 100 / 0 / 0 | 0 / 100 / 0 |
| Experimental | Overlap | 0 / 0 / 100 | 0 / 0 / 100 |

These are paired, fixed-geometry developmental examples, not independent
representative draws or confidence intervals. The overlap is intentionally
severe and may hide an object: metadata truth describes the underlying scene,
not necessarily what a human can recognize in its rendered pixels. Thus the
abstentions are not established classification errors; they demonstrate that
neither checker can assess these examples. In particular this gives no estimate
of the false-accept rate, no general rate for generated images, and no evidence
that a more permissive checker would be valid.

Reproduce using a new path:

```sh
uv run --no-sync python -m exposure.checker_overlap --seed 82995 --pairs 200 \
  --output outputs/checker-overlap-82995.json
```

The ignored result's SHA-256 is
`bc2180e95d991201455dccec7fd1c4ff4afaffc73c3d69777289b3132328ee7b`.
The next checker-validation step is a frozen candidate evaluated on an
untouched, human-labeled generated-image set with explicit negative and
ambiguous cases, class-specific coverage and error targets. Existing reviewer
confirmation of independence should be retained; this stress test does not
invalidate the completed reviews.
