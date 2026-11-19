# Protecting the unwritten read during a sparse memory update

On 24 previously used VitaminC development pages, I tested whether protecting the head's response to the unwritten matrix reduces unintended verdict changes. The weight-one solver adds the native normalized unwritten read as a target while it optimizes the written contrast. An interpolation control moves toward the original query until its unwritten-read displacement matches the protected query as closely as possible. All three methods use the same 256-coordinate write and four query positions. The run reproduces the archived original solver's verdict logits at all 96 endpoints.

| Query rule | Correct on 86 changed | Correct on 10 stable | Lost of four previously correct stable verdicts | Unwritten verdict agrees with old /96 |
|---|---:|---:|---:|---:|
| Original normalized read | 50 | 3 | 3 | 91 |
| Protected unwritten read | 34 | 3 | 2 | 93 |
| Matched displacement along original direction | 37 | 3 | 2 | 94 |

The pilot missed the prespecified joint criterion for changed answers, preservation and sham stability. Across 384 query positions, the median absolute difference in unwritten-read displacement between the protected and interpolation rules was 3.8e-08 (maximum 0.00389).

[Protocol](contract.json) · [Recomputed results](pilot/summary.json) · [Paired outputs](pilot/rows.jsonl) · [Matching audit](pilot/paired_checks.json) · [Frozen file hashes](frozen_files.json) · [Unmodified sparse write](../verdict_write)

## Reproduction

From the repository root, use the pinned four-bit Qwen checkpoint, the original VitaminC development JSONL, MLX on Apple Silicon, NumPy, SciPy and a C++ compiler with Apple's Accelerate library. Supply fresh output and build directories.

```sh
bash experiments/quantized_write/build.sh
python experiments/guarded_read/experiment.py "$QWEN9B" . "$VITAMINC_DEV" "$SCRATCH" "$BUILD"
python experiments/guarded_read/score_guard.py "$SCRATCH"
```
