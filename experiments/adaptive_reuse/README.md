# Choosing which donor MLP positions to refresh

In each of the first 28 layers I measured the squared difference between the two post-normalized inputs to the MLP. The drift arm refreshed the largest differences at the fixed token budget. The anchored arm also protected the two edits, their next three token positions, and the donor read boundary. The upper-layer control restored all donor MLP computations in the final four layers before the write. All arms used the same 16-step readout. I evaluated eight previously exposed seeds and eight additional development seeds, both text formats, and four write/control conditions. The comparison charges both encodings, both branches, the write, the readout and the first full-vocabulary answer.

| Schedule | Written targets / 32 | Preservation / 96 | Mean complete time (s) | Time relative to joint | Identical donor writes to joint / 128 |
|---|---:|---:|---:|---:|---:|
| Two-row batched prefill | 28 | 96 | 0.4719 | 1.000 | 128 |
| Shared prefix, all tokenwise projections | 28 | 96 | 0.3918 | 0.830 | 0 |
| Refresh changed neighborhoods and tail | 27 | 96 | 0.3478 | 0.737 | 0 |
| Refresh largest normalized changes | 27 | 96 | 0.3494 | 0.740 | 0 |
| Protect edit neighborhoods, then use drift | 28 | 96 | 0.3520 | 0.746 | 0 |
| Restore the last four MLP layers | 28 | 96 | 0.3579 | 0.758 | 0 |

The rule selected the protect edit neighborhoods, then use drift schedule for independent testing.

Timing is paired by seed within each experiment; device conditions vary between runs. The archived score includes per-reference timing ratios and paired intervals. Inputs, solver weights and the zero-based intervention sites are pinned.

[Protocol](contract.json) · [Raw outputs](pilot/rows.jsonl) · [Recomputed results](pilot/summary.json) · [Frozen source hashes](frozen_files.json)

## Reproduction

Use the pinned [four-bit Qwen model](../read_projection/release.json), MLX on Apple Silicon, NumPy, SciPy and the compiled reader in the repository. Use a fresh scratch output directory.

```sh
bash experiments/quantized_write/build.sh
python experiments/adaptive_reuse/run_adaptive.py "$QWEN9B" . pilot "$SCRATCH"
python experiments/adaptive_reuse/score_adaptive.py pilot "$SCRATCH"
```
