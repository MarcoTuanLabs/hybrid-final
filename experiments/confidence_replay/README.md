# Replaying uncertain sparse writes

The previous independent schedule lost a reference success on a close verdict. This follow-up uses the gap between the two largest full-vocabulary logits to decide whether to repeat the entire request with shared-prefix computation. Both attempts count toward latency. I fixed one threshold, an edit-aware primary rule, a fixed-position router, two full-computation references and disjoint confirmation prompts. The pilot determines whether to run the 32 new seeds. [Protocol](contract.json) · [Confirmation prompt hashes](prompt_hashes.json) · [Earlier lost target](../write_schedule_confirmation)

## Development pilot

| Schedule | Target answers /16 | Controls /48 | Reference successes lost | Mean seconds | Replays /64 |
|---|---:|---:|---:|---:|---:|
| Two-row prefill | 12 | 48 | 0 | 0.4449 | 0 |
| Shared prefix | 12 | 48 | 0 | 0.3770 | 0 |
| Fixed neighborhoods | 12 | 48 | 0 | 0.3323 | 0 |
| Neighborhoods and drift | 12 | 48 | 0 | 0.3261 | 0 |
| Neighborhoods and drift, with replay | 12 | 48 | 0 | 0.3499 | 3 |
| Fixed neighborhoods, with replay | 12 | 48 | 0 | 0.3473 | 3 |

The primary rule passed the registered criterion. Its time ratio to two-row prefill was 0.787 (95% paired seed-bootstrap interval 0.740 to 0.848). Its time ratio to shared prefix was 0.928 (95% paired seed-bootstrap interval 0.877 to 0.992). On the written targets alone, the primary/reference time ratios were 0.948 and 1.102, respectively. Replaying changed 0 target verdicts from incorrect to correct and 0 from correct to incorrect. [Complete rows](pilot/rows.jsonl) · [Scoring](pilot/summary.json)

## Independent confirmation

| Schedule | Target answers /64 | Controls /192 | Reference successes lost | Mean seconds | Replays /256 |
|---|---:|---:|---:|---:|---:|
| Two-row prefill | 53 | 192 | 0 | 0.5384 | 0 |
| Shared prefix | 52 | 192 | 1 | 0.4536 | 0 |
| Fixed neighborhoods | 53 | 192 | 1 | 0.3927 | 0 |
| Neighborhoods and drift | 52 | 192 | 1 | 0.3968 | 0 |
| Neighborhoods and drift, with replay | 52 | 192 | 1 | 0.4242 | 12 |
| Fixed neighborhoods, with replay | 52 | 192 | 1 | 0.4185 | 13 |

The primary rule missed the registered criterion. Its time ratio to two-row prefill was 0.788 (95% paired seed-bootstrap interval 0.758 to 0.820). Its time ratio to shared prefix was 0.935 (95% paired seed-bootstrap interval 0.900 to 0.971). On the written targets alone, the primary/reference time ratios were 0.937 and 1.088, respectively. Replaying changed 0 target verdicts from incorrect to correct and 0 from correct to incorrect. [Complete rows](confirm/rows.jsonl) · [Scoring](confirm/summary.json)

All 12 independent replays occurred on the 64 written targets. Across all 64 written targets, the router took 8.8% longer on average than direct shared-prefix execution. The one lost reference success triggered a replay at a 0.25-logit gap. The shared-prefix path still favored the recipient by 0.25 logits, while the two-row path favored the donor by 0.25. Substituting the archived two-row output on the 12 triggered requests would recover that target. This retrospective comparison has no independent timing measurement. [Paired fallback diagnostic](fallback_diagnostic.json)

Confidence-based computation has precedents in early exits and model cascades. Here the two paths are the approximate and full donor computations within the same four-bit memory intervention. [CALM](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract-Conference.html) · [FrugalGPT](https://arxiv.org/abs/2305.05176)

I fixed the source, threshold and independent prompt hashes at [registration 1350c259](https://github.com/MarcoTuanLabs/hybrid-final/tree/1350c259cc4207524e32386caddb7413faac28ed/experiments/confidence_replay). For full measurement, use the pinned four-bit weights, the compiled solver, MLX on Apple Silicon, NumPy and SciPy. Supply a fresh scratch directory.

```sh
bash experiments/quantized_write/build.sh
python experiments/confidence_replay/route.py "$QWEN9B" . pilot "$SCRATCH"
python experiments/confidence_replay/score_route.py pilot "$SCRATCH"
```

## Projection work

The pilot router used 23.5% fewer dense-projection multiply-accumulates than two-row prefill and 8.7% fewer than shared-prefix computation. [Counts](pilot/projection_cost.json) The confirm router used 23.5% fewer dense-projection multiply-accumulates than two-row prefill and 8.2% fewer than shared-prefix computation. [Counts](confirm/projection_cost.json) These estimates exclude the recurrent update, attention score products, vocabulary projection, routing and data movement.

An expanded cohort audit recovered 540 prior or pilot seed values. None of the 128 confirmation prompt triples matched any of their 2,160 prompt triples. [Cohort audit](cohort_audit.json)
