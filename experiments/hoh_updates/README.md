# Answering questions after Wikipedia revisions

I selected one question per page from the public HoH-QAs release. Retrieval indexed 111,971 distinct evidence pairs from 90,571 pages. Each pair contains the page title and one earlier and later piece of evidence. The index contains no benchmark questions or reference answers. I fixed 32 development pages before generation and reserved 96 pages for transfer. After the title exclusions, 90,549 pages remained eligible for evaluation. The release labels its examples as training data. These splits describe this experiment. [Dataset and dated snapshots](https://huggingface.co/datasets/russwest404/HoH-QAs/tree/8f12fd356240fa9386d20346fe1e1f9b928d1b64) · [HoH paper](https://aclanthology.org/2025.acl-long.301/) · [Selection](selection.json)

Both variants rank passages using a shared dense ranking. The direct list draws from the top 32 current BM25 results. The expanded list can also return candidates found in older evidence or pseudo-feedback search. Both hybrid retrieval variants placed the labeled current evidence in the top four on 30/32 questions. Both the direct four-passage Qwen3.5-4B reader and the current-only RoBERTa span reader returned 23 exact answers. The direct generator had token F1 .821 and the span reader .840. Expanding retrieval across both editions yielded 22 exact answers from Qwen. Showing it both editions together yielded 20, including four answers that matched only the earlier reference. The current-only readers gave zero earlier-only answers.

| Method | Exact / 32 | Token F1 |
|---|---:|---:|
| Direct four-passage Qwen reader | 23 | .821 |
| Expanded four-passage Qwen reader | 22 | .798 |
| Expanded Qwen reader shown both editions | 20 | .755 |
| Whole-cache contrast during generation | 22 | .804 |
| Attention-cache contrast during generation | 22 | .804 |
| Recurrent-matrix contrast during generation | 22 | .798 |
| Current-only span reader, expanded four | 23 | .840 |
| Project the most likely old span | 19 | .770 |
| Project the old span distribution | 22 | .832 |

Cache decoding used the current state as the positive branch. On each of the first six greedy steps, the negative branch used the entire old cache, the old full-attention cache, or the old recurrent matrices. Both branches took the same generated tokens. The span experiment aligned subtoken sequences and transported probability mass from the 64 highest-scoring old spans to the current passage. Relative to the current-only span reader, this changed one exact answer into an error. Neither native contrast recovered an additional exact answer. No candidate reached the prespecified three-question improvement over all practical controls. [Frozen comparison](4b_contract.json) · [All 13 methods, stale answers and paired changes](summary.json)

I scored the latest short answer against the release's single reference, ignoring capitalization, punctuation and English articles. An answer can give the correct facts with different wording and lose exact-match credit. The retrieval corpus consists of the release's evidence excerpts. The original HoH setup uses an LLM grader and different retrieval collections. Qwen's seven policies shared some prefill work during this accuracy experiment, so the archived times describe the experimental run. [Generated answers](gen_rows.jsonl) · [Span scores and outputs](span_rows.jsonl)

## Reproduction

Python 3.11 or later can rescore the frozen outputs without any model. Run from the repository root.

```sh
python experiments/hoh_updates/score.py
```

For fresh inference, copy the scripts to a new directory as below. The programs write intermediate files beside themselves. The dataset card specifies Apache 2.0 for HoH-QAs, and identifies Wikipedia evidence as CC BY-SA 4.0. The public artifact includes predictions, retrieval identifiers and question-level labels. The builder reconstructs the versioned source passages from the pinned release. [Dataset card and attribution](https://huggingface.co/datasets/russwest404/HoH-QAs/blob/8f12fd356240fa9386d20346fe1e1f9b928d1b64/README.md) · [Wikipedia license](https://creativecommons.org/licenses/by-sa/4.0/)

```sh
work=$(mktemp -d)
mkdir "$work/hoh_updates" "$work/bridge_reads"
cp experiments/hoh_updates/*.py experiments/hoh_updates/*contract.json experiments/hoh_updates/selection.json "$work/hoh_updates/"
cp experiments/remastered_paths/bridge_reads/engine.py "$work/bridge_reads/"
python "$work/hoh_updates/prepare.py" /path/to/hoh_qas_240601_241201.parquet
python "$work/hoh_updates/rank.py" /path/to/contriever-msmarco
python "$work/hoh_updates/span.py" /path/to/roberta-base-squad2
python "$work/hoh_updates/generate.py" /path/to/Qwen3.5-4B
python "$work/hoh_updates/score.py"
```

The builder uses PyArrow. The ranker and extractive reader use PyTorch, Transformers and NumPy on CPU. Qwen inference used MLX 0.32.4 and mlx-lm 0.32.2 on Apple silicon. [4B model files](../late_arrival/model_4b.json) · [Retrieval weights](https://huggingface.co/facebook/contriever-msmarco/tree/abe8c1493371369031bcb1e02acb754cf4e162fa) · [Extractive weights](https://huggingface.co/deepset/roberta-base-squad2/tree/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4)
