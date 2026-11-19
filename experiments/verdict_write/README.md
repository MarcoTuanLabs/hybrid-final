# Can the two-head input write change a real factual verdict?

I used all 24 earlier VitaminC development pages, both versions of both claims, and the original natural-language prompts. I applied the same layer-28, two-head 256-coordinate donor write before the four known chat tokens. The sham designed the same queries and used the unmodified memory. The all-recurrent and all-attention transfers locate effects in larger cache components. I scored both the model’s highest vocabulary token and its best legal A/B/C verdict. Reused caches make this a causal diagnostic. No service latency was measured. [Original page split](../vitaminc_read/split.json) · [VitaminC study](https://aclanthology.org/2021.naacl-main.52/)

The 96 endpoints contain 86 changes and 10 stable judgments. On 44 changed endpoints the model's unedited read of each original document gave the correct legal answer.

| Reader | Legal changed | Legal stable | Both-native-correct changed | Full-vocabulary correct, all endpoints |
|---|---:|---:|---:|---:|
| Old document | 11 | 4 | 0 | 15 |
| Full donor document | 62 | 4 | 44 | 66 |
| Two-head write, native query | 11 | 4 | 0 | 15 |
| Two-head write, whitened key | 43 | 3 | 27 | 46 |
| Two-head write, normalized readout | 50 | 3 | 33 | 53 |
| Normalized query, unwritten sham | 10 | 4 | 0 | 14 |
| All donor recurrent and convolution states | 61 | 5 | 43 | 66 |
| All donor attention caches | 15 | 5 | 1 | 20 |

The sparse readout missed the development criterion and received no independent confirmation.

[Protocol](contract.json) · [Raw outputs](pilot/rows.jsonl) · [Recomputed results](pilot/summary.json) · [Frozen source hashes](frozen_files.json)

## Reproduction

Use the pinned [four-bit Qwen model](../read_projection/release.json), MLX on Apple Silicon, NumPy, SciPy and the compiled reader in the repository. Use a fresh scratch output directory.

```sh
bash experiments/quantized_write/build.sh
python experiments/verdict_write/run_verdict.py "$QWEN9B" . "$VITAMINC_DEV" "$SCRATCH"
python experiments/verdict_write/score_verdict.py . "$SCRATCH"
```

Both unedited document reads were correct together on 0/10 stable endpoints. For all four previously correct stable verdicts, the unedited donor gave the wrong verdict. The normalized sparse write lost three of the four, matching the donor’s error in each case. [Stable judgments by paired verdict](pilot/stable_pairs.json)
