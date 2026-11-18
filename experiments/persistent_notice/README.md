# One document update across three turns

I updated one cache before any question arrived. The same conversation then asked about the changed fact, an unaffected fact and the changed fact in new words. The model saw its own answers on later turns. The twelve development documents included six bAbI stories and six MQuAKE fact lists. I used the second published question phrasing for MQuAKE and a fixed location paraphrase for bAbI. Each question used the same reference aliases as its corresponding first phrasing. The basis was fitted on twelve different documents. All evaluation documents had been used during earlier development. [Frozen design](contract.json) · [Question phrasings](paraphrases.json) · [MQuAKE](https://github.com/princeton-nlp/MQuAKE)

The native patch projects a recurrent matrix difference onto a rank-four basis before the first question. The matrix-only version keeps the old attention and convolution states. The full-bank version also transmits their changes. For the matched retrieval arms, the receiver applies the exact same text diff, appends a notice of what changed, and retrieves up to four current statements per turn. The sender sees only the document. The fresh-retrieval control starts without an old model cache. [Service implementation](serve.py)

Each count below has a denominator of twelve documents. Joint accuracy requires the changed answer to be correct in both phrasings. Three-turn agreement means every generated token sequence matched a fresh replay on the corresponding turn. The criterion for factual correctness was exact first-line agreement with a normalized reference alias. Cold time includes the original ingestion, update and all three replies. Warm time assumes both the original cache and any required updated donor are ready. Both include a rate-capped local socket emulating 1 Gbit/s. Serving times exclude model loading. The one-time basis installation was timed separately.

| Policy | Changed | Unaffected | Rephrased | Both changed answers correct | All three sequences match replay | Cold mean, s | Warm mean, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old cache | 0 | 8 | 0 | 0 | 0 | 1.248 | 0.957 |
| Old cache + notice | 11 | 8 | 11 | 11 | 9 | 1.545 | 1.239 |
| Old cache + notice + retrieval | 12 | 8 | 12 | 12 | 7 | 1.704 | 1.402 |
| Fresh retrieval | 9 | 8 | 10 | 9 | 1 | 1.358 | 1.358 |
| Complete replay | 11 | 8 | 10 | 10 | 12 | 1.622 | 1.294 |
| Checkpoint replay | 11 | 8 | 10 | 10 | 12 | 1.544 | 1.214 |
| 16-token replay | 11 | 8 | 10 | 10 | 12 | 1.575 | 1.262 |
| Recurrent patch | 0 | 8 | 0 | 0 | 1 | 1.543 | 1.004 |
| Recurrent patch + notice + retrieval | 12 | 8 | 12 | 12 | 7 | 1.963 | 1.450 |
| All-bank projection | 10 | 8 | 9 | 9 | 10 | 1.435 | 0.964 |

The notice with retrieval answered both changed questions for all twelve documents. On the same inputs, adding the recurrent patch changed no answer's correctness on any of the 36 turns. It increased mean cold service time from 1.704 to 1.963 seconds. Its cold-time ratio relative to full replay was 1.211 (paired 95% document bootstrap interval 1.116 to 1.311). The matrix-only patch without text answered zero changed questions correctly. The full replay answered both phrasings correctly on ten documents and preserved eight unaffected answers, underscoring the difference between reproducing a replay and answering correctly.

The preregistered accuracy gate required two more jointly correct documents than the strongest cheap baseline. Its observed 12/12 score placed that margin beyond the maximum possible here. Every unchanged suffix was short enough that the 16-token policy skipped zero tokens. This sample offers no evidence about a genuinely partial replay or about retention beyond three turns. The basis cost 786,491 bytes and 0.011 seconds to install once. Raw outputs include logical cache size, wire bytes and per-stage times. [Full results and paired wins and losses](summary.json) · [Raw observations](rows.jsonl) · [Installation](install.json)

## Reproduction

The standard-library scorer uses the published reference aliases. To rerun inference, follow the shared-basis experiment to regenerate its installation file from the pinned model. Use a disposable checkout with an empty observations file for fresh timing. [Shared-basis instructions](../shared_read_basis/README.md) · [Code and evidence hashes](bundle.json)

```sh
p=experiments/persistent_notice
python "$p/score.py" .
python experiments/shared_read_basis/fit.py "$QWEN9B" .
python "$p/serve.py" "$QWEN9B" .
```
