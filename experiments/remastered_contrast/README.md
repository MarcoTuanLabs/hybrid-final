# Which cache channel helps with edited facts?

On two 512-edit banks, a Qwen3.5-4B reader that amplified the change attributable to recurrent memory answered 54 of 96 questions correctly. A conventional scoped reader answered 55. Both answered 10 of the 24 multi-hop questions affected by edits. The recurrent contrast required more decoder work and missed the criterion for a larger run.

| Reader | Edited single / 24 | Edited multi / 24 | Unaffected single / 24 | Unaffected multi / 24 | Total / 96 | Decoder token positions |
|---|---:|---:|---:|---:|---:|---:|
| all_context | 14 | 10 | 12 | 7 | 43 | 6,661 |
| adaptive | 24 | 10 | 14 | 7 | 55 | 4,394 |
| dense | 24 | 5 | 13 | 9 | 51 | 5,836 |
| paired | 24 | 10 | 14 | 7 | 55 | 5,487 |
| whole | 24 | 8 | 14 | 6 | 52 | 7,923 |
| native | 24 | 10 | 14 | 6 | 54 | 7,895 |

The unconditional reader runs on the retrieved current facts for every question. The other readers use a shared conventional shortcut for recognized single-relation questions. An explicit subject–relation match returns its stored object. A recognized question outside the scope of the updates receives a short factual prompt. The remaining 49 questions use the model and retrieved facts. The shortcut accounts for the improvement on single-hop edits. Each reader independently generated its answers or completed the registered direct lookups.

Among the 24 edited multi-hop questions, the adaptive retrieval contained every required update for 16. Its reader answered ten of those correctly. It missed all eight questions where at least one update was absent. This retrospective diagnostic uses benchmark path annotations only after decoding. [Evidence coverage](retrieval_diagnostic.json)

The adaptive reader uses Contriever scores, explicit subject matches and graph adjacency to choose a prefix of ranked evidence. The dense reader takes its top four text matches. The paired prompt presents previous and current facts together. The whole-prompt contrast amplifies differences between outputs conditioned on current and old facts. The recurrent contrast fixes the current attention cache and convolution history, then replaces the pre-question recurrent matrices with the ones built from the old facts for its contrast branch. Both contrastive decoders use the same weight and plausible-token mask. Training-free contextual contrast has precedent. [Context-aware decoding](https://aclanthology.org/2024.naacl-short.69/) · [Question construction and attribution](ATTRIBUTION.md)

This is a balanced development experiment on a derived workload. A shared retriever processed 17,376 padded document positions while building the two indices. The decoder column excludes retrieval. [Fixed protocol](original_contract.json) · [Raw answers](rows.jsonl) · [Computed scores](summary.json)

## Reproduction

From the repository root, run the CPU scorer. It verifies that each reader has one answer for every question.

```sh
python experiments/remastered_contrast/score.py
```

The archive includes the exact banks, rankings and 36 public question templates used at inference. For fresh answers, install MLX and mlx-lm on Apple silicon, set the model path and run the reader. To rebuild the rankings, use a separate environment with PyTorch and Transformers 4.57.1 and supply the Contriever model directory for revision abe8c1493371369031bcb1e02acb754cf4e162fa.

```sh
HYBRID_QWEN_WEIGHTS=/path/to/Qwen3.5-4B python experiments/remastered_contrast/readers.py
python experiments/remastered_contrast/score.py experiments/remastered_contrast/fresh_rows.jsonl
HYBRID_CONTRIEVER_WEIGHTS=/path/to/contriever python experiments/remastered_contrast/rank.py
```
