# Fusing a known suffix across a native write

I appended the four known read tokens before computing the recipient's lower 28 layers. At layer 28 I split the prefix from the suffix, transferred the donor's 256 convolution values, and solved the same four constrained queries. I then processed the resulting recipient sequence through the upper layers. A second arm computed only the last donor query at layer 27 and the last recipient query at layer 31. All methods returned the first token from the full vocabulary. A persistent generation cache would require a separate service measurement.

| Schedule | Written targets / 16 | Preservation / 48 | Mean complete time (s) | Time relative to joint | Identical donor writes to joint / 64 |
|---|---:|---:|---:|---:|---:|
| Separate truncated donor | 13 | 48 | 0.4624 | 1.092 | 0 |
| Two-row batched prefill | 14 | 48 | 0.4236 | 1.000 | 64 |
| Fused suffix | 14 | 48 | 0.3672 | 0.867 | 0 |
| Fused suffix, terminal-query pruning | 13 | 48 | 0.3604 | 0.851 | 0 |

The primary schedule missed the development rule.

Timing is paired by seed within each experiment; device conditions vary between runs. The archived score includes per-reference timing ratios and paired intervals. Inputs, solver weights and the zero-based intervention sites are pinned.

[Protocol](contract.json) · [Raw outputs](pilot/rows.jsonl) · [Recomputed results](pilot/summary.json) · [Frozen source hashes](frozen_files.json)

## Reproduction

Use the pinned [four-bit Qwen model](../read_projection/release.json), MLX on Apple Silicon, NumPy, SciPy and the compiled reader in the repository. Use a fresh scratch output directory.

```sh
bash experiments/quantized_write/build.sh
python experiments/write_window/experiment.py "$QWEN9B" . pilot "$SCRATCH"
python experiments/write_window/score.py pilot "$SCRATCH"
```
