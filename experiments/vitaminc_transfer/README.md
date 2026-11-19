# Independent check of the specialist result

On 44 development pages the hybrid model's paired readout missed its advancement rule. In an exploratory control on the same pages, a paired readout of the published ALBERT-xlarge verifier answered 73/88 endpoints against 67/88 for the original classifier. This independent experiment tests whether that specialist result holds on new pages and against simpler paired classifiers. [Development comparisons](../vitaminc_joint/pilot/summary.json) · [Fixed criteria](contract.json)

I registered the implementation, sampling rule, comparisons and 224 test-page hashes before running the model on those pages. All 240 calibration pages come from the official training split. All evaluation pages come from the publisher's test split and are disjoint from the training file and the 468 pages previously fitted, inspected or reserved. The stratified workload has 448 factual verdicts. Its score applies to those selected natural revision pairs. [Frozen case IDs](split.json) · [VitaminC dataset](https://huggingface.co/datasets/tals/vitaminc/tree/be6febb761b0b2807687e61e0b5282e459df2fa0)

## Results

The readout answered 373/448 endpoints correctly, compared with 365/448 for the original specialist. The strongest alternative, contrast of the two native logits, answered 372/448. The paired reader gained 14 individual verdicts and lost 6 against the original. It answered 32/56 stable endpoints correctly against 31/56 for the original. Its smallest accuracy gain across the six fixed comparisons was 0.22 percentage points (95% page-bootstrap interval -2.01 to 1.56). The registered decision was **unconfirmed**. Endpoint pairs share a page and claim. Each bootstrap draw resampled the 224 pages and recomputed the minimum comparison. [Exact counts and registered decision](confirmation/summary.json) · [Per-page outputs](confirmation/rows.jsonl)

| Reader | Correct / 448 | Changed / 392 | Stable / 56 | Both editions / 224 |
|---|---:|---:|---:|---:|
| Paired type and orientation head | 373 | 341 | 32 | 154 |
| Published ALBERT-xlarge | 365 | 334 | 31 | 146 |
| Supervised single-edition regression | 369 | 338 | 31 | 150 |
| Separate mean and difference heads | 367 | 336 | 31 | 149 |
| Contrast of the two native logits | 372 | 343 | 29 | 155 |
| Calibrated label-transition prior | 370 | 340 | 30 | 152 |
| Nine-class joint regression | 370 | 338 | 32 | 152 |

Calibration reused the logits from 240 earlier training pages. At inference the paired heads use the two editions’ specialist logits. Case selection and code were public before inference. [Registration at fba83140](https://github.com/MarcoTuanLabs/hybrid-final/tree/fba83140f8773d5371f35d273c61eb7ebfda74fb/experiments/vitaminc_transfer)

Among the 92 support/refute pages the paired reader scored 158/184 endpoints versus 163/184 for the original. On the 92 support/insufficient pages it scored 167/184 versus 159/184. The reader’s relative performance varied with the kind of revision. [All six label pairs](confirmation/by_label_pair.json)

## Reproduction

Recompute every prediction, comparison and interval from the archived logits with NumPy. Use a scratch copy for the output. The optional model run additionally needs the pinned publisher training and test JSONLs, the ALBERT-xlarge checkpoint, PyTorch, Transformers and SentencePiece.

```sh
p=experiments/vitaminc_transfer
mkdir -p "$SCRATCH"
cp "$p/confirmation/logits.jsonl" "$SCRATCH/logits.jsonl"
python "$p/analyze.py" "$SCRATCH"
# To regenerate the logits from the source records
python "$p/transfer.py" run "$RAW_DIR" --model "$MODELS/albert-xlarge-vitaminc" --output "$SCRATCH"
```
