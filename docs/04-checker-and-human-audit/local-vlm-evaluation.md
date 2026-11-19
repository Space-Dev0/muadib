# Local vision-language model as a candidate checker

This experiment evaluates a local vision-language model as a possible
alternative checker for the previously reviewed target-color-circle question.
It does not assume the model is correct, replace the existing independent
reviews, validate relational predicates, or establish Stage 0 passage.

The first candidate is Qwen2.5-VL 3B via Ollama on the RTX 3050 laptop.
Qwen3-VL 4B was attempted first, but the installed Ollama server rejected its
manifest as requiring a newer version. The 3B model and its multimodal support
are described in the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
and the [Ollama model library](https://ollama.com/library/qwen2.5vl). The
experiment is local: the API connection is loopback through an SSH tunnel, and
the evaluation tool rejects non-loopback URLs.

Before the model sees the images, `exposure.vlm_audit prepare` fixes the image
order, source hashes, target color, exact prompt, structured-output schema,
nearest-neighbor 64-to-256 pixel enlargement and decoding options. Human
answers and checker predictions are not included in model requests. The
first load attempt with Ollama's default batch exhausted available memory;
a second attempt using `num_batch` 64 also exceeded available memory. No model
answers were produced by either attempt. A smaller
[Moondream 1.8B](https://ollama.com/library/moondream) is being evaluated with
the same frozen question and batch size. The
model's exact Ollama digest is recorded before inference; changing it, the
plan, or the image hashes prevents resume. Unparseable outputs are `unsure`,
not silently counted as correct. Individual responses are stored so a
completed run can be resumed without repeating them.

On the frozen 64-image comparison, Moondream answered yes for every image:
58/58 human-positive images and 6/6 human-negative images. Thus it falsely
accepted all six negatives and is not suitable as a replacement checker on
this sample. All 64 outputs were well-formed, so this was not an abstention
or parsing failure. Ollama reported the model as 100% GPU. Its digest was
`55fc3abd386771e5b5d1bbcc732f3c3f4df6e9f9f08f1131f9cc27ba2d1eec5b`;
the ignored plan and score hashes were
`4dea23ce6c38b84a51f2a64bfb937cc6ce0a52d72dffd839d29ee179b7a322c8` and
`1252453f6241a5a283ae33a239b4f1cf80a1ee24605134572c5f436737e79869`.

After that result, a Qwen preflight with smaller context and forced GPU-layer
settings loaded successfully, though Ollama reported it as 100% CPU. Its
free-form description of previously reviewed item `matched-0003` was used for
development, so this item is excluded from the subsequent frozen 63-image
comparison. The Qwen model and its prompt/settings must be kept distinct from
the Moondream evaluation.

On the remaining 63 images from that block, Qwen answered yes for all 58
human-positive images and unsure for all five human-negative images. There
were no human-unresolved cases in that block. All five unsure responses were
valid structured model outputs, not parsing failures. Thus the model avoided
false accepts in this small sample by making no negative decisions; its
accepted-case false-accept rate is undefined because it accepted no negative
cases. The model digest was
`fb90415cde1ef08aa669ae74b082d49b158729b6db1ab183c941417d507e71a1`;
the ignored plan and score hashes were
`4481b2478adb07d8febdb54d5166c286876e89f00a67b88bd32cb21ee248ee5d` and
`0881027ef06a74672c9e8e9c6a7b1a684b6ba487b7af8a734b0d830210b42ad4`.

On the separately frozen first matched-model block, Qwen answered yes on all
59 human-positive images and unsure on all three human-negative images. Of
the two human-unresolved images, Qwen answered yes to one and no to one; the
available reviews cannot say which, if either, was correct. Thus, across both
blocks, Qwen answered none of the eight human-negative cases definitively.
The first-block plan and score hashes are
`f5595ae23e60eab8f2139daeae5825744a734feed8a72912001fde3ae93e3195` and
`460af8319c4c02dc86250f587d440966d6846efcd285bcd7a2472b0d50b685c6`.

An additional exploratory candidate asks Qwen to list visible objects and
their color–shape pairs, without giving it the requested target color. Code
then answers yes for a matching object, unsure when detection is ambiguous,
and no otherwise. Both reviewed blocks were frozen for this inventory-based
comparison before inspecting its outputs. Because this candidate was devised
after seeing the direct-question results, the comparison is developmental;
it is not independent confirmation or a new human audit.

The inventory candidate also answered yes for all 58 human-positive cases and
unsure for all five human-negative cases in the first 63-image comparison.
Inspection showed the negative inventories did not list the requested circle
but did set the model's uncertainty flag. A post-hoc ablation ignores that
flag when every listed object's color and shape are known; it still abstains
for malformed or unknown attributes. On these same 63 cases the ablation
matched 58 positives and five negatives. This is a development result chosen
after seeing the first block, not validation of the ablation or evidence that
ignoring uncertainty is generally safe. The second block was already running
when the ablation rule was recorded (eight responses had been generated but
not inspected); its comparison must also be described as exploratory.
The second inventory block matched 59 human-positive cases and abstained on
all three human-negative cases; it answered yes on one human-unresolved case
and abstained on the other. The post-hoc ablation matched all 59 positives and
all three negatives, and answered yes/no on the two unresolved cases. Across
both blocks it matched 117 human-positive and eight human-negative cases; two
human-unresolved cases cannot contribute to correctness. This is encouraging
but far below the negative-case sample needed for the illustrative 2% error
bound and does not validate relations. The first/second ablation output hashes
are `5e0c5041cb369c96fd782f45a2564b8d8d0c11ea710157e75fe5553a5745415a`
and `a358ce6d1ce3b7d4141568da5b34ba34e20a3e394744a655ab1f02dad4fa2dc3`.

A further blind inventory run on all 150 red-object and square-object
questions from the earlier 800k pilot is in progress. It uses the same
post-hoc object-list ablation, with a frozen prompt and no target or human
label sent to the model. It excludes the 150 relation questions because this
inventory has no spatial coordinates. Most of these 150 images have only one
reviewer; their labels are not an independent consensus. This is another
exploratory check, not a pass of the model-checker or Stage 0 gates.

The comparison uses the four completed independent-review files from the
second 64-image matched-seed block. The scorer rechecks their provenance and
CSV hashes and reports the full truth-by-output table, including abstentions
and human-unresolved cases. Only six cases were human-negative in this block;
even perfect agreement cannot by itself establish a 2% false-accept rate.
The small sample and shared training dataset also cannot establish general
validity, inter-model uncertainty, or performance on other rules and image
distributions. If this candidate performs well, a separately frozen larger
evaluation remains necessary before trusting it on new research outcomes.

The protocol can be reproduced with a locally forwarded Ollama port and fresh
output paths:

```sh
uv run --no-sync python -m exposure.vlm_audit prepare \
  --pack outputs/mosaic-independent-review-500k-83000 --output outputs/vlm-audit-new
uv run --no-sync python -m exposure.vlm_audit run \
  --pack outputs/mosaic-independent-review-500k-83000 --output outputs/vlm-audit-new \
  --url http://127.0.0.1:11435 --model qwen2.5vl:3b
uv run --no-sync python -m exposure.vlm_audit score \
  --pack outputs/mosaic-independent-review-500k-83000 --output outputs/vlm-audit-new
```
