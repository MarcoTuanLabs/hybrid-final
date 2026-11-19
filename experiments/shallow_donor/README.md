# Predicting a deep memory update from layer 12

The exact donor in the four-bit 9B sparse-write service can stop at the layer-28 input projection. I tested whether 12 decoder layers suffice to predict the 256 values needed for a write. One regression predicts the deep donor slot from the shallow donor activations. The paired method predicts how the shallow activations change between documents and adds its prediction to the recipient's exact deep slot. Both methods use the existing normalized 16-step reader and the original four suffix positions. [Frozen design](contract.json) · [Exact truncation](../jacobian_service/README.md)

The predictors use four vectors per prompt from the final token, the mean of the final eight, and the two token positions at which recipient and donor differ. A fixed signed Hadamard projection retains 384 coordinates per vector. The absolute ridge regression uses a fitted intercept. The paired regression maps zero change to zero and uses a separate RMS scale for each feature block. The 192 calibration pairs came from 48 fresh seeds. Only model activations entered the regressions. Calibration took 66.30 s on the Apple M5. Both fitted models together occupied 3,160,096 bytes. The costs below cover tokenization, both branches, prediction or exact extraction, native writing, read optimization and the full-vocabulary answer. Model loading and calibration precede timing.

| Method | Controlled targets | Correct controls | Mean elapsed time, s | Median cosine to exact write displacement |
|---|---:|---:|---:|---:|
| Exact truncated donor | 13/16 | 48/48 | 0.5664 | 1.000 |
| Full batched donor and recipient | 14/16 | 48/48 | 0.5079 | 0.989 |
| Shallow prediction of the absolute donor slot | 6/16 | 37/48 | 0.4314 | 0.428 |
| Original deep slot plus predicted shallow difference | 3/16 | 48/48 | 0.4253 | 0.326 |

The paired predictor retained all 48 controls but lost 10 target successes relative to exact truncation, with 0 gains. Its time ratio was 0.751 (95% paired seed-bootstrap interval 0.741 to 0.762) against truncation and 0.837 against full two-row batching. The eight development seeds had been used for the exact-reader benchmark. The registered promotion criterion failed. The confirmation seeds remain unused. [Raw service measurements](dev_rows.jsonl) · [All estimates](dev_summary.json)

Affine probes of intermediate hidden states have a precedent in the tuned lens. This experiment tests whether shallow paired differences predict a deep *memory write* that remains effective at the model output. [Belrose et al.](https://arxiv.org/abs/2303.08112)

## Reproduction

The archived score uses Python's standard library. A fresh run requires Apple Silicon, the model pinned in the reader release and its compiled solver. Use a disposable checkout, move archived phase rows aside, then regenerate the small ridge models before timing. The calibration receipt records the original weights digest and serving parameter size. The reported experiment uses inference-time estimated writes.

```sh
p=experiments/shallow_donor
python "$p/score.py" dev
bash experiments/quantized_write/build.sh
python "$p/reader.py" calibrate "$QWEN9B" .
python "$p/reader.py" dev "$QWEN9B" .
python "$p/score.py" dev
```

[Model pins](../read_projection/release.json) · [Calibration receipt](calibration.json) · [Artifact hashes](bundle.json)
