# Blinded human audit workflow

This tool prepares a review manifest and scores **actual reviewer judgments**
against separately supplied checker predictions. It does not generate human
labels, send images to anyone, or declare a research gate passed. The proposal's
audit of at least 300 generated images remains to be performed.

## Prepare a review sample

Create a JSON file with a `samples` list. Each sample identifies an image file
relative to the dataset root, its requested rule, and the rule's predicate and
relation counts. For example:

```json
{
  "samples": [{
    "sample_id": "sample-001",
    "image_path": "images/sample-001.png",
    "rule_name": "red",
    "predicates": [{"type":"attribute","variable":"a","field":"color","value":"red"}],
    "k_attr": 1,
    "k_rel": 0
  }]
}
```

Use unique sample IDs. All records with the same rule name must use the same
predicate list, and complexity counts must match the predicates. The command
checks that each path resolves to a file within the dataset root, including
after resolving symlinks. It samples the requested number from **each stratum
present**; it fails if any present stratum has too few images. Missing strata
are not invented or declared covered. The output cannot overwrite the input
manifest or any referenced image, including through symlink or hardlink aliases.

```sh
uv run python -m exposure.human_audit prepare \
  --manifest outputs/audit-candidates.json --dataset-root outputs/generated \
  --per-stratum 20 --seed 42 --output outputs/audit-blinded.json
```

The resulting manifest contains only IDs, relative image paths, rule names and
predicates, and complexity counts; additional source fields such as checker
predictions or gold labels are omitted. Supply the manifest and images to the
reviewer through an appropriate workflow. Reviewers should independently judge
whether each image satisfies its requested rule. Use neutral IDs and filenames:
the tool cannot tell whether a user-provided name itself reveals a prediction.

## Score completed reviews

The current prototype checker can produce a separate predictions file from the
review manifest and the image files:

```sh
uv run python -m exposure.human_audit predict \
  --manifest outputs/audit-blinded.json --dataset-root outputs/generated \
  --output outputs/checker-predictions.json
```

This reads pixels and evaluates each requested rule; it writes only a JSON
mapping of IDs to `true`, `false`, or `null` for a detector rejection. An image
loading error aborts prediction; it is not counted as a detector rejection. The
prototype checker is not validated on diffusion outputs. Keep predictions out of
the material shown to the reviewer. This command does not create judgments or
modify the blinded manifest; its output path must differ from the manifest and
all referenced image files.

Save actual reviewer decisions in a CSV with precisely these columns:

```csv
sample_id,judgment
sample-001,yes
```

Judgments must be `yes` or `no` (case-insensitive). The separately produced
checker predictions are a JSON object mapping each reviewed sample ID to the
checker decision: `true` for satisfied, `false` for not satisfied, or `null`
for a detector rejection. A separately implemented checker may supply this same
format instead.

```json
{"sample-001": true}
```

```sh
uv run python -m exposure.human_audit score \
  --manifest outputs/audit-blinded.json --reviews outputs/reviews.csv \
  --predictions outputs/checker-predictions.json --output outputs/audit-results.json
```

The reviewer CSV and prediction JSON must each contain exactly the manifest's
IDs; duplicates and malformed values are rejected. The output cannot overwrite
the manifest, review CSV, or prediction JSON, including through symlink or
hardlink aliases. The report includes overall,
per-rule, and per-complexity counts of true positives, true negatives, false
positives, false negatives, and rejections split by human yes/no judgment.
False-accept and false-reject rates use **only non-rejected images** as their
respective human-no and human-yes denominators. A missing denominator produces
`null`. Inspect rejection counts alongside these conditional error rates;
neither rates nor counts make claims about unreviewed images.

Each command writes its result to a temporary file in the output directory and
then replaces the requested output. A failed validation or write leaves any
existing output file intact.

## Separate predicate questions

Whole-rule judgments cannot establish which individual predicates were true.
After preparing the complexity-stratified rule manifest, generate separate
questions for each predicate in every selected rule:

```sh
uv run python -m exposure.human_audit predicates \
  --manifest outputs/audit-blinded.json --dataset-root outputs/generated \
  --output outputs/audit-predicates.json
uv run python -m exposure.human_audit predict \
  --manifest outputs/audit-predicates.json --dataset-root outputs/generated \
  --output outputs/predicate-predictions.json
uv run python -m exposure.human_audit score \
  --manifest outputs/audit-predicates.json --reviews outputs/predicate-reviews.csv \
  --predictions outputs/predicate-predictions.json --output outputs/predicate-results.json
```

Collect an independent `yes` or `no` judgment for each question in
`predicate-reviews.csv`, with the same `sample_id,judgment` columns used for
whole rules. The checker predictions must remain hidden from reviewers. Each
predicate is evaluated on its own with existential object binding, just as in
the exposure calculations; its objects need not be those that satisfy another
predicate in the full rule. A full rule can therefore be false even when every
separately evaluated predicate is true. Do not infer these judgments from the
whole-rule answers.

The report's `by_rule` keys identify predicate events such as
`attribute:color:red` and `relation:above`. The manifest retains
`source_sample_id` and a zero-based `predicate_index` to trace each question to
the original selected rule. Use neutral source IDs and filenames. Multiple
questions can refer to the same image, so question counts are not counts of
independent images. The predicate report's strata describe the singleton
questions, not the original rule complexity; use the whole-rule report for
the original strata. Each review CSV contains one judgment per question; the
scorer does not assign reviewers, track reviewer IDs, resolve disagreement, or
calculate agreement or a research gate. Review the same questions in separate
files if multiple reviewers are needed, and agree on a reconciliation protocol
before treating any combined judgments as final.
