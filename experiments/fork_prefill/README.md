# Sharing the common prefix in a hybrid prefill

I stored identical prefix tokens once for the first 28 layers. I expanded the hidden states to two ordinary rows for causal attention or gated-delta updates. The packed arm shared all tokenwise projections. The MLP arm shared only the feed-forward computation. Both kept the complete terminal layer. In exact arithmetic, causal states for the identical prefix agree. The four-bit kernels may round differently when their row layouts change. [Hydragen](https://arxiv.org/abs/2402.05099) previously shared attention computation across prefixes. This implementation tests the tokenwise work around a recurrent-memory intervention.

| Schedule | Written targets / 16 | Preservation / 48 | Mean complete time (s) | Time relative to joint | Identical donor writes to joint / 64 |
|---|---:|---:|---:|---:|---:|
| Two-row batched prefill | 14 | 48 | 0.5073 | 1.000 | 64 |
| Fused suffix | 14 | 48 | 0.4405 | 0.868 | 0 |
| Shared prefix MLP | 14 | 48 | 0.4269 | 0.841 | 0 |
| Shared prefix, all tokenwise projections | 14 | 48 | 0.4222 | 0.832 | 0 |

The primary schedule missed the development rule.

The small FP32 block check covers three branch points and four native blocks. It compares outputs to ordinary execution and changes only donor inputs to check that recipient states remain fixed. [Causal indexing check](causal_check.json)

The operation estimate counts logical dense weight-matrix multiply-accumulates. It excludes attention scores, recurrent updates, solver arithmetic, vocabulary projection and device traffic. [Projection counts](pilot/projection_cost.json)

Timing is paired by seed within each experiment; device conditions vary between runs. The archived score includes per-reference timing ratios and paired intervals. Inputs, solver weights and the zero-based intervention sites are pinned.

[Protocol](contract.json) · [Raw outputs](pilot/rows.jsonl) · [Recomputed results](pilot/summary.json) · [Frozen source hashes](frozen_files.json)

## Reproduction

Use the pinned [four-bit Qwen model](../read_projection/release.json), MLX on Apple Silicon, NumPy, SciPy and the compiled reader in the repository. Use a fresh scratch output directory.

```sh
bash experiments/quantized_write/build.sh
python experiments/fork_prefill/run_fork.py "$QWEN9B" . pilot "$SCRATCH"
python experiments/fork_prefill/score_fork.py experiments/fork_prefill/contract.json "$SCRATCH" pilot
```
