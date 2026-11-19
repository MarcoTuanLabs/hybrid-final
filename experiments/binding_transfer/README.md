# Transferring the fixed reader to the CAPITALS binding task

I tested the original layer-28 native write on an externally designed binding task. The CAPITALS benchmark names several people and the countries whose capital cities they live or work in. A question asks for the city associated with one person. I used the original DEFAULT and WORK completion templates, the authors' first-name and country/capital tables, and their seeded permutation order. I applied the single-token filter using the pinned Qwen tokenizer, yielding 98 names and 52 country/capital entries. Each of 12 fixed prompt IDs supplies three people and two templates. A target donor exchanges the queried person's country with another. A nuisance donor exchanges the unqueried pair. [Feng and Steinhardt, ICLR 2024](https://arxiv.org/abs/2310.17191) · [Pinned benchmark source](https://github.com/jiahai-feng/binding-iclr/tree/fb7608dd4c255ac37d7e95c5c8bf45b5512901eb/coref/datasets)

The last four tokens of the authors' completion prefix serve as the four read slots. I copied the original layer-28 value-history coordinates into the recipient and measured the top token over the whole vocabulary. The higher-budget native control copied every value channel of the same newest convolution slot. The opportunity column includes only pairs on which the complete original and donor prompts each produced the intended capital. All prompts contribute to the first and third columns. The margin is the alternate capital's logit minus the original capital's logit on changed cases. [Fixed design](contract.json)

| Input or reader | Changed capital | Changed, both clean prompts correct | Correct nuisance answer | Median target margin |
|---|---:|---:|---:|---:|
| Original prompt | 0/24 | 0/18 | 20/24 | -10.53 |
| Full donor prompt | 20/24 | 18/18 | 19/24 | 10.36 |
| Native query, 256-value write | 0/24 | 0/18 | 20/24 | -10.56 |
| Native query, 4,096-value write | 0/24 | 0/18 | 20/24 | -10.47 |
| Whitened key, 256-value write | 0/24 | 0/18 | 20/24 | -10.47 |
| Normalized reader, 256-value write | 0/24 | 0/18 | 20/24 | -11.31 |

The full donor returned the intended alternate capital on 20/24 targets. The fixed sparse write controlled 0/24 with normalization and 0/24 with state-Gram whitening. Across the 72 nuisance and unwritten presentations, the normalized reader changed 1 native top token. The task uses named capital completions and a natural-language suffix. It tests the transfer of the complete fixed configuration, including its write site and placement. The preregistered transfer criterion failed and the 64 confirmation IDs remain unopened for model inference. [All controls and comparisons](dev_summary.json) · [Per-prompt model outputs](dev_rows.jsonl)

## Reproduction

The source loader verifies the authors' pinned CSVs and prompt templates. It reads the string constants from the template file as data. The standard-library scorer uses the 48 archived prompt pairs. For fresh inference use a disposable checkout, move the archived rows aside and supply the locally pinned MLX model. No model was trained for this transfer.

```sh
p=experiments/binding_transfer
python "$p/score.py" dev
python "$p/inputs.py" "$BINDING_DATA"
bash experiments/quantized_write/build.sh
python "$p/probe.py" dev "$BINDING_DATA" "$QWEN9B" .
python "$p/score.py" dev
```

[Model release](../read_projection/release.json) · [Artifact hashes](bundle.json)
