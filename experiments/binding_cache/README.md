# Which cache entries carry an updated association?

The fixed answer-time write from the letter task failed to transfer to the public CAPITALS task. I moved the edit to the end of the document before presenting any question. In each of 24 documents I exchanged the countries of two people, then asked about all three people independently from the same cache. I used both original templates and all 12 previously evaluated development prompt IDs. Updating the complete donor cache is an attainable behavioral reference. A cache can produce the correct capital with a different set of successes from that reference. [Task and source](../binding_transfer/README.md) · [Frozen comparison](contract.json)

I changed only the named components in each donor-based control. The four local rows are the two changed country tokens and their following punctuation in each of the eight full-attention layers. Each row includes all four KV heads. Recurrent states and convolution histories stayed at their old values for the attention interventions. The two last controls require only the original document and the requested change. The three independent questions include both changed people and the person whose country stayed fixed.

| Updated component | Correct changed capital | Correct stable capital | Changed successes lost vs. full | Changed successes gained vs. full | All top tokens matching full |
|---|---:|---:|---:|---:|---:|
| Old document | 0/48 | 19/24 | 40 | 0 | 25/72 |
| Complete updated document | 40/48 | 19/24 | 0 | 0 | 72/72 |
| All updated recurrent and convolution states | 0/48 | 19/24 | 40 | 0 | 27/72 |
| All updated attention K/V | 38/48 | 19/24 | 2 | 0 | 68/72 |
| Updated K/V at the two country tokens | 37/48 | 19/24 | 3 | 0 | 67/72 |
| Updated K/V at countries and following punctuation | 40/48 | 19/24 | 2 | 2 | 66/72 |
| Updated values at those four positions | 40/48 | 18/24 | 1 | 1 | 67/72 |
| Updated keys at those four positions | 0/48 | 19/24 | 40 | 0 | 25/72 |
| Swap the two pairs of old values | 27/48 | 18/24 | 13 | 0 | 53/72 |
| Append the two corrected statements | 28/48 | 15/24 | 12 | 0 | 55/72 |

Transplanting the full donor's four local K/V rows answered 40/48 changed questions correctly. It gained 2 changed successes and lost 2 against complete replay. Its stable top tokens all matched the old model's. Replacing all recurrent states alone controlled 0/48 changed capitals. Reassembling all donor cache components reproduced the complete donor's vocabulary logits on all 72 questions. The local condition passed the prespecified criterion for testing a donor-free estimator. [Paired results](summary.json) · [All outputs](rows.jsonl)

The donor-based conditions already have the complete updated cache. Their execution tests causal sufficiency and shares prefills across methods. For these short documents, the cost of original ingestion and three future questions was 2,376 model positions. Replaying the suffix from the earliest changed token adds 432 positions. Even a free edit would remove only 15.4% of positions in this workload. This calculation covers token positions and excludes checkpoint construction and kernel effects.

Prior work tests entity binding through activation interchange and cache patching separates K/V contributions. These controls locate changed associations for the pinned hybrid checkpoint on this specific benchmark. [Binding ID](https://arxiv.org/abs/2310.17191) · [KV-cache patching](https://arxiv.org/abs/2610.41480)

## Reproduction

Use the archived scorer with standard Python. For new inference run the pinned Qwen decoder on Apple Silicon, using the upstream records retrieved by the neighboring task. Move archived rows out of a disposable checkout before running inference.

```sh
p=experiments/binding_cache
python "$p/score.py"
python experiments/binding_transfer/inputs.py "$BINDING_DATA"
python "$p/run.py" "$BINDING_DATA" "$QWEN9B" .
python "$p/score.py"
```

[Model assets](../read_projection/release.json) · [Artifact hashes](bundle.json)
