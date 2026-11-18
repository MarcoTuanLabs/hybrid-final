# Which revision supports the answer?

I used the expert-written questions from DiffQG to test factual reading across Wikipedia revisions. The original benchmark asks models to generate questions. Here I asked a model to answer a released question from a bank of 350 forward-year passage pairs. Each request supplied the question and requested snapshot year. I retained 32 development topics and reserved 96 distinct topics. Retrieval used the full contexts after removing the annotations marking the edited sentence. Neither the answer span nor the target page entered the inference program. The release also contains backward-year examples. I excluded those before generating any answers. [Original dataset and paper](https://aclanthology.org/2023.eacl-main.225/) · [Data and original evaluation](https://github.com/google-research/language/tree/master/language/diffqg) · [Selection](selection.json) · [Direction correction](schema_correction.json)

BM25 and Contriever ranked evidence for the requested year. The first passage contained the labeled source on all 32 requests. I therefore included a control that answered directly from the first passage. Across retrieved passages, an extractive reader proposed spans and the algorithm located lexical edits with sequence alignment. For each of two relevant edit regions, I reverted that region in the newer text and inserted it into the older text. I tested whether support for a proposed answer fell on rollback and rose on insertion. All open readers used Qwen3.5-4B. [Method contract](contract.json) · [Scoring equations in words](rule_details.json) · [First-passage comparator](top1_control.json)

| Method | Exact / 32 | Token F1 |
|---|---:|---:|
| Extractor, first passage | 22 | 0.849 |
| Extractor, six passages | 22 | 0.849 |
| Prefer spans overlapping a change | 20 | 0.843 |
| Compare two complete editions | 22 | 0.863 |
| Undo one edit | 23 | 0.884 |
| Undo and reapply one edit | 22 | 0.849 |
| Free QA, first current passage | 20 | 0.845 |
| Free QA, three current passages | 22 | 0.879 |
| Free QA, six current passages | 21 | 0.869 |
| Free QA, three paired editions | 21 | 0.860 |
| Free QA with ordinary span proposals | 21 | 0.857 |
| Free QA with changed-span proposals | 23 | 0.880 |
| Free QA with whole-edition proposals | 22 | 0.859 |
| Free QA with rollback proposals | 23 | 0.880 |
| Free QA with two-direction proposals | 21 | 0.858 |

The union of proposed spans contained an exact reference answer on 31/32 cases. For the ordinary first-passage extractor, seven of the ten exact misses still supplied the requested fact in the judgment of one unblinded AI reviewer. One gave only the year instead of the part of the model year. Two answered incorrectly. [Per-case review](top1_reference_review.json) · [Complete summary and paired differences](summary.json)

The strongest control scored 23/32 exact and the highest control F1 was 0.880. No method met the three-answer improvement gate with preserved F1. The confirmation topics remained unopened.

## Reproduction

Run the commands from the repository root in an unused working copy. This expands the archived CPU outputs. The standard-library scorer then reproduces the summary. Its timings describe shared development calls.

```sh
p=experiments/diffqg_hunks
gzip -dc "$p/span_rows.jsonl.gz" > "$p/span_rows.jsonl"
python "$p/score.py"
```

To rerun inference, download the official gold annotations, then use the scripts below in a fresh copy containing the sibling experiment directories. Remove the archived append-only open predictions from that fresh copy before generating. The prepared corpus and split have frozen hashes. No model reads the label or focal-sentence files while answering. [Official data release](https://storage.googleapis.com/gresearch/diffqg/gold_annotations.jsonl) · [Pinned reader and retrieval weights](../hoh_updates/README.md#reproduction) · [Generation code](../hoh_inverse/README.md)

```sh
p=experiments/diffqg_hunks
python "$p/prepare.py" /path/to/gold_annotations.jsonl
python "$p/retrieve.py" "$CONTRIEVER"
python "$p/read.py" "$ROBERTA_QA"
python "$p/answer.py" "$QWEN4B"
python "$p/score.py"
```
