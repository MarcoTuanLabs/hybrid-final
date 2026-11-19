# Predicting two factual verdicts from the change in evidence

On the [first pilot](../vitaminc_read), a paired linear readout gained three correct endpoints over a direct supervised classifier while losing two net correct stable endpoints. I widened the evaluation to single claims, removed the compact-edit requirement, and included natural revisions involving the neutral verdict. I tested whether estimating the two possible verdicts jointly can preserve unaffected claims. [VitaminC study](https://aclanthology.org/2021.naacl-main.52/) · [Protocol](contract.json)

I selected one claim per Wikipedia page. I fitted on 240 pages from the official training split and evaluated 44 new pages from the official development split. All 284 pages are distinct and disjoint from the earlier 184 exposed or reserved pages. The 44-page sample contains 36 changed labels and eight stable labels. Each passage is treated once as current. All 88 verdicts count. I stratified by the pair of publisher labels before inference. The resulting distribution represents this revision workload. The corpus-wide score and deployment prevalence require separate estimates. [Split](split.json)

The reader uses both normalized Qwen layer-16 states for the same claim. One head estimates the probabilities of the six unordered label pairs from the mean state and the absolute half-difference. For each distinct-label pair, a second head uses the signed half-difference to assign the labels to the two passages. I marginalize these probabilities before choosing each verdict. Reversing the input order reverses the predictions. Comparators include a single-edition classifier, a linear paired classifier, full Qwen with both editions explicitly identified in the prompt, and two published task-trained ALBERT classifiers. I also trained the same readouts on each specialist's three logits. The old gold label is unavailable to every method.

| Reader | Correct | Changed verdict | Stable verdict | Both editions correct |
|---|---:|---:|---:|---:|
| Reuse old Qwen verdict | 14 | 10 | 4 | 2 |
| Full Qwen reread | 52 | 48 | 4 | 19 |
| Explicit prompt showing both editions | 55 | 49 | 6 | 18 |
| Full-logit contrast | 53 | 49 | 4 | 20 |
| Qwen direct supervised head | 64 | 51 | 13 | 23 |
| Qwen mean and difference heads | 56 | 49 | 7 | 17 |
| Qwen label-pair and orientation heads | 65 | 54 | 11 | 24 |
| Published ALBERT-base | 66 | 57 | 9 | 26 |
| ALBERT-base logits with label-pair head | 67 | 58 | 9 | 26 |
| Published ALBERT-xlarge | 67 | 57 | 10 | 28 |
| ALBERT-xlarge logits with label-pair head | 73 | 62 | 11 | 31 |

The structured Qwen reader answered 65/88 endpoints correctly, including 11/16 stable endpoints. The strongest comparator scored 73/88. The advancement rule required four additional correct endpoints over the strongest comparison and at most one net stable loss against the direct Qwen head. It missed the development rule and was dropped from scale-up. Timings from full-pass feature capture are excluded from deployment claims. [Every comparison](pilot/summary.json) · [Individual predictions](pilot/rows.jsonl)

In a post-hoc diagnostic, the readout identified the correct unordered pair of labels on 28/44 pages. When given that pair, it assigned the two labels to the correct passages on 32/36 changed claims and scored 80/88 endpoints overall. This diagnostic uses information unavailable during inference. [Breakdown by label pair](pilot/diagnostic.json)

## Reproduce

Use the model and library setup in the [earlier pilot](../vitaminc_read). Here `RAW_DIR` contains both pinned JSONL splits named `train.jsonl` and `dev.jsonl`. `CACHE` is a new scratch directory. The public evidence contains hashes, labels and predictions. Obtain the passages from the publisher.

```sh
p=experiments/vitaminc_joint
for part in fit pilot; do
    python "$p/collect.py" "$RAW_DIR" "$QWEN" "$CACHE" experiments/read_projection/release.json "$part"
done
for model in albert-base-vitaminc-mnli albert-xlarge-vitaminc; do
    python "$p/special.py" "$RAW_DIR" "$MODELS/$model" "$CACHE/scores"
done
python "$p/structured.py" "$CACHE" "$CACHE/scores"
```
