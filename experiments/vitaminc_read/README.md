# Reading a factual verdict after the evidence changes

I tested whether a cached verdict from one version of a Wikipedia passage helps a hybrid model judge the same claim against another version. I used the publisher's three labels and Qwen3.5-9B in four-bit precision. Each endpoint asks about one passage. I treated both editions in turn as current. This paired design makes no chronological assumption. [VitaminC, Schuster et al. (NAACL 2021)](https://aclanthology.org/2021.naacl-main.52/) · [Pinned release](https://huggingface.co/datasets/tals/vitaminc/tree/be6febb761b0b2807687e61e0b5282e459df2fa0)

I calibrated on 64 pages and evaluated 24 different pages in the official development split. Each page supplied two claims with unambiguous publisher labels in both editions. The 96 endpoints contained 86 whose label changed and ten whose label stayed fixed. Only three endpoints had the “not enough information” label. The split includes 96 additional page IDs reserved before this experiment. The reserved pages received no inference. The passages and claims are available from the publisher. [Split and case hashes](split.json) · [First contract](contract.json)

For the first candidate I fitted a ridge map from the difference between normalized layer-16 states to the difference in the full Qwen label logits. At inference I added that predicted change to the stored previous logits. I used the previous model output without access to its gold label. The trained classifier control used the publisher's gold labels for the 64 calibration pages. All model predictions used the same three possible labels.

| Reader | Correct | Changed verdict | Stable verdict | Both editions correct |
|---|---:|---:|---:|---:|
| Reuse previous Qwen verdict | 15 | 11 | 4 | 0 |
| Full Qwen reread | 66 | 62 | 4 | 22 |
| Prior verdict plus learned change in teacher logits | 63 | 56 | 7 | 21 |
| Direct middle-layer teacher probe | 66 | 60 | 6 | 21 |
| Direct middle-layer label classifier | 79 | 72 | 7 | 32 |
| Label-transition prior from the old verdict | 69 | 63 | 6 | 24 |
| Contrast the full verdict logits | 69 | 64 | 5 | 25 |

The teacher-difference reader scored 63/96 against 66/96 for full replay. The second candidate trained separate linear heads for the average and signed difference of paired verdict labels. It gained three endpoints over the direct supervised classifier and lost two net stable endpoints. Its fixed advancement rule required at least four additional correct endpoints and at most one net stable loss. Both pilots ended before independent confirmation or service timing. [Second contract](component_contract.json) · [First outcome rows](pilot/rows.jsonl) · [Second outcome rows](pilot/component_rows.jsonl)

| Reader | Correct | Changed verdict | Stable verdict | Both editions correct |
|---|---:|---:|---:|---:|
| Direct middle-layer classifier | 79 | 72 | 7 | 32 |
| Separate mean and difference heads | 82 | 77 | 5 | 37 |
| Replace only the difference head | 80 | 73 | 7 | 33 |
| Replace only the mean head | 83 | 78 | 5 | 36 |
| Published ALBERT-base, VitaminC and MNLI | 77 | 71 | 6 | 31 |
| Published ALBERT-xlarge, VitaminC | 81 | 75 | 6 | 33 |

The number of pages is the independent sample count. The scores measure a selected paired workload. They do not estimate official VitaminC accuracy. The full-pass feature collector did not time a deployed early-exit path. The published ALBERT results use the authors' evidence-then-claim order. [Base model](https://huggingface.co/tals/albert-base-vitaminc-mnli/tree/5c9fea6e2ffa1fe7187fc3d29ffeba5d77b46da9) · [Xlarge model](https://huggingface.co/tals/albert-xlarge-vitaminc/tree/e86807a8bfa4f9cfec822c5855f1c86c4f475f03)

## Reproduce

Use the pinned model files, VitaminC development JSONL, MLX on Apple Silicon, and PyTorch with SentencePiece for the ALBERT controls. Set `RAW` to the development JSONL, `QWEN` to the local [Qwen checkpoint](../read_projection/release.json), `MODELS` to the parent of both ALBERT model directories and `CACHE` to a scratch directory. Run from the repository root.

```sh
p=experiments/vitaminc_read
for part in fit pilot; do
    python "$p/run.py" "$RAW" "$QWEN" "$CACHE" "$part" experiments/read_projection/release.json
done
python "$p/score.py" "$CACHE" pilot "$CACHE/scores"
for model in albert-base-vitaminc-mnli albert-xlarge-vitaminc; do
    python "$p/specialist.py" "$RAW" "$MODELS/$model" pilot "$CACHE/scores"
done
python "$p/decompose.py" "$CACHE" pilot "$CACHE/scores"
```
