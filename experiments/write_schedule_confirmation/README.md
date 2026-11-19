# Independent test of a donor-side MLP schedule

The 16-seed development screen selected `anchored`. I fixed the scoring rule, comparison arms and 32 fresh seeds before confirmation. Each seed supplies two text formats and four write or preservation conditions, for 256 service executions per method. Timing uncertainty resamples seeds. [Locked protocol](contract.json) · [Prompt-pair hashes](prompt_hashes.json) · [Development record](../adaptive_reuse)

## Results on the new seeds

Edit neighborhoods plus activation drift controlled 52/64 written targets and preserved 192/192 controls. It lost one previously successful target against the union of the two full-computation references. Its measured time ratio against two-row prefill was 0.740 (95% paired seed-bootstrap interval 0.735 to 0.746). The ratio to exact shared-prefix computation was 0.884 (0.869 to 0.900). The registered result was **unconfirmed**. [Complete scoring](confirmation/summary.json) · [Individual requests](confirmation/rows.jsonl)

| Schedule | Written targets /64 | Preservation /192 | Mean complete seconds | Mean seconds, written targets alone |
|---|---:|---:|---:|---:|
| Two-row prefill | 53 | 192 | 0.4659 | 0.4645 |
| Shared prefix with full donor computation | 53 | 192 | 0.3904 | 0.3929 |
| Fixed edit neighborhoods and tail | 52 | 192 | 0.3413 | 0.3428 |
| Edit neighborhoods plus activation drift | 52 | 192 | 0.3449 | 0.3424 |
| Activation drift alone | 52 | 192 | 0.3436 | 0.3419 |
| Full donor MLP in last four lower layers | 51 | 192 | 0.3504 | 0.3507 |

On the written targets alone, the candidate/reference time ratio was 0.737 against two-row prefill and 0.871 against packed prefill. The candidate's donor slot exactly matched the two-row reference on 0/256 requests. The write layout and the four-slot read solver were fixed. Reusing donor feed-forward contributions changes the values supplied at those sites. [Per-reference target wins and losses](confirmation/paired_checks.json) · [Logical dense projection costs](confirmation/projection_cost.json)

In the single lost case (seed 88213015, prose), the donor token led the recipient token by 0.125 logits under two-row prefill and 0.25 under shared-prefix execution. With the registered schedule, the recipient token led by 0.5. Drift alone returned exactly the same 64 target decisions as the registered schedule. The identical target counts for the fixed-position and registered schedules hide one win and one loss between them.

I fixed the code, thresholds, comparison arms and exact prompt hashes publicly before starting independent inference. [Registration fdd8007f](https://github.com/MarcoTuanLabs/hybrid-final/tree/fdd8007f2b53aeb095b8499fb835a69d07ff7919/experiments/write_schedule_confirmation)

## Reproduction

Rescore the archived rows in a scratch directory. Fresh measurement requires the pinned four-bit Qwen weights, MLX on Apple Silicon, NumPy, SciPy and the compiled read solver.

```sh
p=experiments/write_schedule_confirmation
mkdir -p "$SCRATCH"
cp "$p/confirmation/rows.jsonl" "$SCRATCH/rows.jsonl"
python "$p/score_confirmation.py" confirm "$SCRATCH"
# For a new run, use a different empty directory
bash experiments/quantized_write/build.sh
python "$p/confirm.py" "$QWEN9B" . confirm "$EMPTY_SCRATCH"
```
