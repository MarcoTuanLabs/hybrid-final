# Answering the same question about two editions

Each of 20 Wikipedia pages contributes one DynaQuest question whose published answer changed between two measured editions. Qwen3.5-4B answers the question twice, once for each date. A page counts as fully correct only when both answers match their respective references. The query date identifies a published snapshot. It makes no additional assertion about when the underlying event took place. [DynaQuest paper](https://aclanthology.org/2025.findings-acl.1380/) · [Sources and the 40 revision links](ATTRIBUTION.md)

## Data and comparators

Nine half-month DynaQuest files spanning August 16, 2024 through January 1, 2025 yielded 2,127 repeated questions with changed answers and usable endpoint revision IDs across 1,854 pages. A further 568 changed question pairs lacked a required endpoint revision ID. A fixed hash specified one question per page, the first 20 development pages and the next 48 reserved pages. All 40 selected Wikipedia revisions were retrieved and checked against their revision IDs. Every selected page had an infobox. The 48 reserved pages stayed outside these experiments. [Selection rule](asof_revision/data_contract.json) · [Source hashes](asof_revision/provenance.json) · [Split record](asof_revision/split.json)

For each edition, the retriever ranked infobox rows and sentences from the first 14 paragraphs. It combined lexical term matches, Contriever scores and a preference for text that changed between the two editions. All seven cache comparisons used the same top five passages per edition. The selected passages contained a whole-word match to the normalized reference on 32/40 questions. The larger set of source rows and paragraphs contained 36/40. Pairing the two retrieved contexts before generation assumes the underlying question is already known. [Retriever](asof_revision/rank.py) · [Evidence coverage](asof_revision/evidence_diagnostic.json)

The model saw earlier evidence first on ten pages and later evidence first on ten. The date-aware text baseline read only the requested edition. A separate RoBERTa extractive question-answerer also received the requested edition. The mixed-context baseline received both dated editions and the query date. The new cache intervention tracked an alternative recurrent matrix for each requested edition during the mixed prefill. In the alternative matrix it skipped the other edition's writes and forgetting, then applied subsequent updates using the original keys, values and retention gates. During decoding it could also mask attention to the other edition. The convolutional state and deeper inputs still followed the original mixed trajectory. This calculation matched a direct recurrence to within (3\times10^{-7}) on 12 fixed-trajectory numerical checks. [Cache implementation](asof_revision/run.py) · [Algebra check](asof_revision/algebra.json)

A second experiment aligned changed fields and sentences across the two source editions. It evaluated direct answers from aligned evidence, a prompt asking for both answers together, and several ways of choosing a proposed answer pair. Every selection method saw the same candidates. Teacher forcing included the answer-ending token. A whole-edition contrast measured whether each candidate fit its own version better than its counterpart. The native variant transferred the change along the next read direction between the two recurrent states after removing the component parallel to the current head read. [Alignment](revision_pairs/align.py) · [Pair scoring](revision_pairs/paired.py)

## Evaluation

| 4B policy | Earlier / 20 | Later / 20 | Both editions / 20 | Answers / 40 |
|---|---:|---:|---:|---:|
| Read both dated editions | 14 | 14 | 14 | 28 |
| Read the requested edition | 12 | 13 | 12 | 25 |
| Extract a span from the requested edition | 8 | 12 | 6 | 20 |
| Mask attention to the other edition | 14 | 13 | 13 | 27 |
| Remove only the other edition’s recurrent writes | 14 | 14 | 14 | 28 |
| Skip the other edition’s entire recurrent transition | 14 | 14 | 14 | 28 |
| Mask attention and skip the other transition | 14 | 13 | 13 | 27 |
| Read aligned fields in the requested edition | 12 | 14 | 11 | 26 |
| Rank answer pairs by ordinary likelihood | 13 | 14 | 12 | 27 |
| Rank pairs by whole-edition contrast | 13 | 14 | 12 | 27 |
| Rank pairs by full recurrent-state contrast | 14 | 14 | 13 | 28 |
| Rank pairs by the normalized read-direction contrast | 13 | 14 | 12 | 27 |

On the mixed-context baseline, the same 14 pages supplied all 28 correct answers. The remaining six pages had two incorrect answers each. The paired proposal pool contained an entirely correct pair on 16 of 20 pages. The promotion criterion required at least 17. Even perfect selection among those proposals would have fallen short. The results favor retaining the ordinary mixed-context reader as a comparator. [Cache study counts](asof_revision/4b_summary.json) · [Candidate-pair study counts](revision_pairs/4b_summary.json)

The source evaluator's answer normalization and token F1 were applied to every date. Plain normalized exact match is recorded separately. The evaluation also counts earlier editions correct, later editions correct and pages with both correct. The published programs recorded actual shared decoder positions and elapsed time. They reused prompts and subcalls across policies, so those times describe the experimental workload. The code for fetching inputs, ranking evidence and generating fresh answers accompanies the per-page records.

## Where the version distinction appeared

The diagnostic supplied each reference answer to the model for scoring after prefill. Both versions used the same question, with no date in the question or prompt header. Each version retained its separately retrieved source text. The comparison exchanged the full-attention keys and values, the convolutional history and the recurrent matrices between the two caches. For all eight combinations, it measured the later-versus-earlier answer log odds, using mean token log probability including the answer terminator. Averaging a component's marginal effect over all six exchange orders gives its Shapley contribution to the change in log odds. The attention component includes its donor's positional offsets. This analysis is conditional on the two caches and known answer strings. [Diagnostic implementation](asof_attribution/probe.py) · [Cross-size comparison rule](asof_attribution/9b_transfer.json)

| Model and subgroup | Pages | Correct joint assignment | Both forced choices correct | Both answers freely generated | Median attention contribution | Median convolution contribution | Median recurrent contribution |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4B BF16, all pages | 20 | 17 | 16 | 11 | 6.910 | 0.293 | 0.292 |
| 4B BF16, six pages missed by the 4B mixed reader | 6 | 3 | 2 | 0 | 0.126 | 0.009 | -0.016 |
| 9B four-bit, all pages | 20 | 19 | 16 | 13 | 5.858 | 0.636 | 0.336 |
| 9B four-bit, six pages missed by the 4B mixed reader | 6 | 5 | 2 | 0 | 0.409 | 0.022 | 0.006 |

A positive joint assignment means the combined log score favors assigning the old answer to the old edition and the new answer to the new edition. Each endpoint may still favor the same answer when scored independently. On the six pages missed by the original 4B mixed reader, 9B favored the correct assignment on five, favored both answers independently on two, and freely generated both on zero. On zero of those six pages did the recurrent contribution and the full-cache gap both exceed 0.05 nats per token, at either model size. The median attention contribution exceeded the median recurrent contribution on the full pool and on the six error pages at both sizes. These are the same development pages throughout. [4B results](asof_attribution/4b_summary.json) · [9B results](asof_attribution/9b_summary.json)

## Reproduction

Run these commands from the repository root to rescore the archived answers and reconstruct every channel contribution using Python's standard library.

```sh
python experiments/dated_versions/check.py
python experiments/dated_versions/check_channels.py 4b
python experiments/dated_versions/check_channels.py 9b
```

The inference wrapper restores frozen inputs in a new directory for each phase. The 4B snapshot study used BF16 weights, the 9B comparison used a four-bit model, and generation used MLX 0.32.4 and mlx-lm 0.32.2 on Apple silicon. The optional ranking, alignment and span phases use PyTorch and Transformers. [4B model manifest](../late_arrival/model_4b.json) · [9B model manifest](../late_arrival/model_9b.json) · [Contriever checkpoint](https://huggingface.co/facebook/contriever-msmarco/tree/abe8c1493371369031bcb1e02acb754cf4e162fa) · [Extractive model](https://huggingface.co/deepset/roberta-base-squad2/tree/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4)

```sh
script=experiments/dated_versions/reproduce.py
python "$script" prepare-only --out /tmp/dated-check
python "$script" erase --model /path/to/Qwen3.5-4B --out /tmp/dated-erase
python "$script" pairs --model /path/to/Qwen3.5-4B --out /tmp/dated-pairs
python "$script" probe4b --model /path/to/Qwen3.5-4B --out /tmp/dated-probe4
python "$script" probe9b --model /path/to/Qwen3.5-9B-4bit --out /tmp/dated-probe9
python experiments/dated_versions/check.py --erase-rows /tmp/dated-erase/asof_revision/4b_rows.jsonl --pair-rows /tmp/dated-pairs/revision_pairs/4b_rows.jsonl
python experiments/dated_versions/check_channels.py 9b --rows /tmp/dated-probe9/asof_attribution/9b_rows.jsonl
```

The other supported phases are `rank` and `align` with Contriever weights, and `span` with the extractive model weights. The scientific inputs and predictions have individual checksums in the repository release manifest. [Input builder](asof_revision/prepare.py) · [Attribution and data terms](ATTRIBUTION.md)
