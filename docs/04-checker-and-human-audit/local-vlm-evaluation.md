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
model's exact Ollama digest is recorded before inference; changing it, the
plan, or the image hashes prevents resume. Unparseable outputs are `unsure`,
not silently counted as correct. Individual responses are stored so a
completed run can be resumed without repeating them.

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
