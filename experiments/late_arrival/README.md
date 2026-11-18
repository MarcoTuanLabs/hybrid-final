# Updating a cached conversation before the question arrives

On 34 held-out questions from shortened LongMemEval histories, a 4-bit Qwen3.5-9B produced the same answer text as full replay on every question after punctuation normalization. One answer acquired a final period. The reduction in estimated decoder arithmetic was 25.3% over the complete workload. Synchronized elapsed time fell by 19.5% on an Apple M5.

| 9B serving method | Exact reference answers / 34 | Token F1 | Sequences identical to full replay / 34 | Full-model token positions | Dense-equivalent operations, trillions | Measured time, seconds |
|---|---:|---:|---:|---:|---:|---:|
| Full suffix replay | 21 | .726 | 34 | 66,967 | 940.12 | 148.82 |
| Reuse old final recurrent state | 21 | .722 | 33 | 50,103 | 702.48 | 124.50 |
| Correct with each head’s retained fraction | 21 | .726 | 33 | 50,104 | 702.50 | 119.77 |
| Read the three highest-scoring BM25 text blocks | 19 | .667 | 22 | 10,878 | 152.10 | 27.40 |

Each row includes the work that method performs to ingest one original history, insert a delayed session and answer a previously unseen question. For the three cache methods, the update-through-answer totals alone were 78.85, 46.70 and 46.45 seconds, respectively. Calculations include the full model projections, feed-forward layers, attention, recurrence and output head at executed positions. Quantized kernels, memory traffic and lower-order operations affect elapsed time. The arithmetic column estimates dense-equivalent work from tensor shapes. [Registered comparison](retention_splice/contract.json) · [Question-level outputs and timing](retention_splice/rows.jsonl)

## Update rule

The server retains the original end cache and checkpoints before the insertion and 16 tokens into the following text. For each recurrent head it multiplies the scalar retention gates across the remaining 496 tokens during the original pass. At edit time it processes the inserted text and the first 16 following tokens. It subtracts the old boundary matrix from the new one, multiplies the difference by the saved scalar and adds it to the old end matrix. For full attention it appends the original suffix keys and values to the newly processed prefix and adjusts the rotary positions of the reused keys. Generation uses the repaired cache. The question arrives after the cache update.

Storing the scalar products adds 3,072 bytes for 768 heads. Holding the extra warm-boundary checkpoint costs a further 48 MiB for its FP32 recurrent matrices on 9B, plus convolution and attention state as represented by the serving system. That checkpoint matters when many sessions or edit boundaries remain live. Exact prefix caching, suffix KV reuse and short recurrent replay provide the relevant prior comparisons. [Suffix Cache Reuse](https://wai-org.com/blog/clm/) · [Tail-Replay](https://arxiv.org/abs/2608.30310)

Mean first-answer centered-logit error fell from .1910 for reuse of the old state to .1544 for the scalar correction, a 19.2% reduction. The registered test required two more matching sequences or at least 20% lower error against that comparator. Neither condition held. Relative to full replay, the 95% paired intervals for the ratios of estimated lifecycle operations and time were .742–.753 and .784–.826, respectively, resampling the 33 groups of questions with distinct evidence sessions. [Grouped intervals](retention_splice/grouped_intervals.json)

## Data and scoring

The experiment used the official cleaned LongMemEval-S knowledge-update questions, dates, answers and user turns. Evidence annotations define a retrospective insertion. The old history contains earlier evidence sessions followed by up to 512 tokens of subsequent user text. The new history inserts the latest evidence session at its chronological position. This provides a controlled edit location and omits much of the original conversation. Gold answer strings are read only during scoring. Exact matching ignores punctuation and capitalization. Token F1 gives partial credit. The official entire-history task uses a separate semantic grader. [LongMemEval paper](https://arxiv.org/abs/2410.10813) · [Source snapshot](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f) · [Data license](DATA_LICENSE.txt)

Of 50 eligible questions, the first 16 in a fixed hash order formed the 4B development set. Scalar correction reproduced all 16 full-replay sequences. Full transport through the original keys and gates reproduced 15. The remaining 34 questions went into the 9B run after the implementation and criteria were locked. The two groups share no evidence session or identical evidence user turn. Two 9B questions share an evidence session. [Development scores](suffix_transport/4b_summary.json) · [Split audit](retention_splice/split_audit.json)

## Reproduction

The following commands rescore all 232 archived method-by-question answers using Python 3 and NumPy.

```sh
python experiments/late_arrival/suffix_transport/score.py 4b
python experiments/late_arrival/retention_splice/score.py
python experiments/late_arrival/check_cost.py
```

For fresh inference, supply the official cleaned JSON file and the pinned local model files. The runner reconstructs the input from the original source in an empty output directory. MLX 0.32.4 and mlx-lm 0.32.2 were used on Apple silicon. Either phase can run independently. [4B model files](model_4b.json) · [9B model files](model_9b.json)

```sh
python experiments/late_arrival/reproduce.py 4b --dataset /path/to/longmemeval_s_cleaned.json --model /path/to/Qwen3.5-4B --out /path/to/new-4b-run
python experiments/late_arrival/reproduce.py 9b --dataset /path/to/longmemeval_s_cleaned.json --model /path/to/Qwen3.5-9B-4bit --out /path/to/new-9b-run
```

## Appending a delayed dated session

A 4B development experiment compared chronological cache repair with appending a delayed session to the existing cache. All four methods used the same 16 questions, dated user turns and 512-token suffix. Each policy built its own cache. The dated-append variant explicitly told the model to place the late record according to its date. Gold answers were scored after generation. [Frozen comparison](late_append/contract.json) · [Question-level outputs and timing](late_append/rows.jsonl)

| Update policy | Correct / 16 | Token F1 | Exact replay sequences / 16 | Full-model positions | Lifecycle seconds |
|---|---:|---:|---:|---:|---:|
| Replay in chronological order | 12 | 0.756 | 16 | 31,282 | 99.36 |
| Per-head retention correction | 12 | 0.756 | 16 | 23,346 | 78.38 |
| Append using the recorded date | 13 | 0.824 | 12 | 23,110 | 74.35 |
| Append with an explicit timeline instruction | 13 | 0.824 | 12 | 23,414 | 76.24 |

The retention correction used 25.4% fewer model positions than full replay. It reproduced 16 complete replay sequences. Directly appending the dated text recovered 13 reference answers. The registered accuracy criterion required two additional correct answers over each append policy. The correction missed that criterion.

All policies used a 128-token maximum history prefill and a 64 MiB allocator free-cache limit. This complete rerun followed memory pressure during an initial bulk-prefill attempt. The timings include history ingest, the late update and one answer. The 4B cases are the original development questions. [Full scores](late_append/summary.json)

The following commands replay the scores and run fresh 4B inference.

```sh
python experiments/late_arrival/late_append/score.py
python experiments/late_arrival/reproduce.py append --dataset /path/to/longmemeval_s_cleaned.json --model /path/to/Qwen3.5-4B --out /path/to/new-append-run
```
