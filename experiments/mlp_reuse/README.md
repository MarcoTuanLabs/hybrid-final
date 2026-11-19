# Reusing donor feed-forward updates

I retained each donor token's own attention or recurrent residual. For most donor positions I added the recipient's MLP contribution. At the two changed positions, the following three convolution positions, and the twelve tokens ending at the read boundary, I computed the donor MLP normally. The same-budget tail control refreshed only end positions. The zero-refresh control reused every MLP contribution. These approximate trajectories can change the actual 256-coordinate write. [CacheBlend](https://arxiv.org/abs/2405.16444) previously studied selective KV recomputation in RAG. Here the component shared across inputs is the feed-forward residual.

| Schedule | Written targets / 16 | Preservation / 48 | Mean complete time (s) | Time relative to joint | Identical donor writes to joint / 64 |
|---|---:|---:|---:|---:|---:|
| Two-row batched prefill | 14 | 48 | 0.5194 | 1.000 | 64 |
| Fused suffix | 14 | 48 | 0.4495 | 0.865 | 0 |
| Shared prefix, all tokenwise projections | 14 | 48 | 0.4310 | 0.830 | 0 |
| Reuse every donor MLP | 0 | 48 | 0.3406 | 0.656 | 0 |
| Refresh only the tail | 0 | 48 | 0.3776 | 0.727 | 0 |
| Refresh changed neighborhoods and tail | 13 | 48 | 0.3798 | 0.731 | 0 |

The primary schedule missed the development rule.

A mistyped dispatch in the warm-up initially sent the exact packed control to the approximate selector. It produced no timed rows. I repaired the dispatch before the pilot. [Repair record](repair.json)

The operation estimate counts logical dense weight-matrix multiply-accumulates. It excludes attention scores, recurrent updates, solver arithmetic, vocabulary projection and device traffic. [Projection counts](pilot/projection_cost.json)

Timing is paired by seed within each experiment; device conditions vary between runs. The archived score includes per-reference timing ratios and paired intervals. Inputs, solver weights and the zero-based intervention sites are pinned.

[Protocol](contract.json) · [Raw outputs](pilot/rows.jsonl) · [Recomputed results](pilot/summary.json) · [Frozen source hashes](frozen_files.json)

## Reproduction

Use the pinned [four-bit Qwen model](../read_projection/release.json), MLX on Apple Silicon, NumPy, SciPy and the compiled reader in the repository. Use a fresh scratch output directory.

```sh
bash experiments/quantized_write/build.sh
python experiments/mlp_reuse/run_reuse.py "$QWEN9B" . pilot "$SCRATCH"
python experiments/mlp_reuse/score_reuse.py pilot "$SCRATCH"
```
