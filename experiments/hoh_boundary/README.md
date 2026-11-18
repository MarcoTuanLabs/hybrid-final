# Reusing an answer across a document edit

A service answers a question about one known Wikipedia page. When the page changes, it receives the new evidence and must answer the same question again. I store the first answer, align the two passages and transfer the answer through their edits. If alignment fails, the reader runs again. A single threshold rejects transfers in which the longer of the old and proposed answers contains more than 1.5 times the words in the shorter. The Qwen fallback reuses exact prompt checkpoints stored every 32 tokens. The CPU fallback runs the span reader directly. Each timed policy pays for the initial answer, update decision and any second model call. Cold model loading and upstream delivery of the page are outside the measured workload. [Protocol](contract.json) · [Repair rule](guard.py) · [Alignment and fallbacks](../hoh_refresh/repair.py)

## Development

On the 32 development pages, Qwen3.5-4B prefix replay answered 25 questions exactly. Both guarded policies answered 27. The text-only guard used 3,290 model positions and 26.05 seconds, compared with 4,480 and 38.96 seconds for replay. Adding the old attention trace cost 26.00 seconds and returned the same exact count and F1. The two-edition CPU reader answered 28/32 both with full rereading and with the guard. I froze the text-only method for both readers before transfer. [Development scores](dev_summary.json) · [Implementation hashes](transfer_lock.json) · [Transfer decision](test_decision.json)

## Transfer to separate pages

The transfer sample comprises 96 pages selected by a new hash order from the public HoH-QAs release. It excludes all 352 pages assigned to the full-bank development, calibration and transfer splits. It also excludes the registered DynaQuest titles. Within each eligible page I used the same question-selection rule as in development. This is a page-scoped adaptation of the source benchmark. It uses one annotated current answer per question. [Source data and terms](../hoh_updates/README.md#reproduction) · [Page and question selection](prepare_test.py)

| Reader and policy | Correct / 96 | Token F1 | Model positions | Total seconds |
|---|---:|---:|---:|---:|
| Qwen3.5-9B, exact prefix replay | 79 | .9374 | 13,679 | 131.56 |
| Qwen3.5-9B, transport without the length check | 79 | .9318 | 9,294 | 82.06 |
| Qwen3.5-9B, transport with the length check | 80 | .9398 | 9,853 | 87.02 |
| CPU span reader, full rereading | 75 | .8879 | 11,329 | 3.66 |
| CPU span reader, transport without the length check | 76 | .9061 | 5,682 | 1.83 |
| CPU span reader, transport with the length check | 77 | .8952 | 6,400 | 2.07 |

The 9B rule reran Qwen on 13 of the 96 updated passages. Compared with exact prefix replay, it saved 28.0% of decoder positions, an estimated 28.2% of arithmetic and 33.9% of elapsed time. It recovered two exact answers and lost one. The paired 95% bootstrap interval for the difference in exact-match rate runs from −2.1 to +5.2 percentage points. The corresponding interval for elapsed-time savings runs from 30.5% to 36.8%. Against the faster unguarded transfer, the length check added one exact answer and 4.97 seconds.

The CPU rule reread 12 updated passages. It saved 43.5% of positions, estimated arithmetic and elapsed time relative to full rereading. It recovered two exact answers and lost zero. The unguarded CPU policy was still faster and had higher mean token F1, with one fewer exact answer than guarded transport. These comparisons concern repeated questions about short passages on known pages. They do not establish an advantage from changing the hybrid model's internal memory or a large accuracy gain on new open-domain questions. [All per-policy scores and paired intervals](test_summary.json) · [Per-page Qwen outputs](test_qwen.jsonl) · [Per-page CPU outputs](test_cpu.jsonl) · [Arithmetic estimates](test_arithmetic.json)

I also repeated the 9B comparison without retaining any prefix checkpoints. Full rereading took 119.39 seconds and answered 79/96 exactly. Guarded repair took 66.56 seconds and answered 80/96, with 43.0% fewer decoder positions and 44.2% less elapsed time. The program regenerates the complete prompt on a fallback. The experiment uses the same 96 pages as the preceding table. [Protocol for the additional comparison](refresh_without_checkpoints.json) · [Paired results](test_nocache_summary.json)

A post-hoc blinded AI assessment rated the 9B replay answers as semantically adequate on 96/96 pages, and the repairs on 95–96/96. The uncertain response repeated a surname. [Semantic review and all judgments](../hoh_semantic/README.md)

## Reproduction

The archived predictions and timings rescore with Python 3.11 or later using the standard library. Run from the repository root.

```sh
python experiments/hoh_boundary/score_boundary.py dev
python experiments/hoh_boundary/score_boundary.py test
```

For fresh inference, complete the sibling full-bank and page-subscription preparation steps to populate `$work/hoh_updates` and `$work/hoh_refresh`. The data builders require PyArrow. The ranker used by the full-bank task is optional for this experiment. [Build the bank and dev split](../hoh_updates/README.md#reproduction) · [Prepare page subscriptions](../hoh_refresh/README.md) · [9B weights](../late_arrival/model_9b.json)

```sh
mkdir "$work/hoh_boundary"
cp experiments/hoh_boundary/*.py "$work/hoh_boundary/"
study="$work/hoh_boundary"
python "$study/prepare_test.py" /path/to/hoh_qas_240601_241201.parquet
python "$study/bench_cpu.py" /path/to/roberta-base-squad2 "$work/hoh_refresh/inputs.json" "$study/dev_cpu.jsonl"
python "$study/bench_qwen.py" /path/to/Qwen3.5-4B "$work/hoh_refresh/inputs.json" "$study/dev_qwen.jsonl" --native
python "$study/bench_cpu.py" /path/to/roberta-base-squad2 "$study/test_inputs.json" "$study/test_cpu.jsonl"
python "$study/bench_qwen.py" /path/to/Qwen3.5-9B-4bit "$study/test_inputs.json" "$study/test_qwen.jsonl"
python "$study/score_boundary.py" dev
python "$study/score_boundary.py" test
python "$study/estimate.py" test "$study/test_inputs.json" /path/to/Qwen3.5-9B-4bit
```

Elapsed time is measured on an Apple M5 with the model resident in memory, with four CPU threads for RoBERTa. Method order rotates by page. Model positions count tokens actually processed by each reader. Dense-equivalent arithmetic estimates use model dimensions and executed sequence lengths, including Qwen output projections, recurrence and causal attention. The estimates omit tokenization, alignment, small elementwise operations and memory movement. All of these costs enter the timed service. The timing intervals resample pages within this hardware run. The study did not measure energy or run-to-run timing variation. [Cost calculation](estimate.py)

Use `python experiments/hoh_boundary/score_nocache.py` for the additional no-checkpoint comparison. Fresh inference uses `python "$study/without_checkpoints.py" /path/to/Qwen3.5-9B-4bit "$study/test_inputs.json" "$study/test_nocache.jsonl"` followed by `python "$study/score_nocache.py"`.
