# Checker precision planning

The proposed 2% error target needs an agreed estimand and a prespecified
sample design. As an illustrative calculation, suppose the evaluation has
independent, identically distributed accepted images within each human-truth
class, and separately bounds false rejects among accepted positives and false
accepts among accepted negatives. With zero observed errors in a class of size
`n`, the exact one-sided upper binomial bound is `1 - alpha**(1/n)`.

For one class, 149 accepted images with zero errors give an upper bound below
2% at one-sided 95% confidence. To bound both class-specific error rates at
family-wise 95% using a Bonferroni adjustment, the corresponding zero-error
requirement is **183 accepted positives and 183 accepted negatives**. If errors
occur, more accepted examples are needed to support the same bound. These
figures are not power guarantees and do not determine how many images to
generate: detector abstentions, truth prevalence and unresolved human labels
all affect the total needed.

The calculation applies only to the selected accepted cases. It says nothing
about errors on abstentions, performance on other rules or conditions, or
variation between trained models. Paired questions about the same image,
reviewers labeling the same image, repeated seeds and clustered model outputs
are not independent images or independent trained models. Dependence,
stratification and human adjudication require a separately justified analysis;
the simple binomial interval must not be presented as valid automatically.
Neither the illustrative 2% target nor this sample size has been approved as
the complete Stage 0 checker gate.

```sh
uv run --no-sync python -m exposure.checker_precision \
  --target 0.02 --family-alpha 0.05 --classes 2
```
