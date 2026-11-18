# Finding an answer in a changed Wikipedia passage

I asked whether the difference between two editions helps a reader answer a new question. I reused the 32 development questions from the full-bank HoH experiment. Each request supplied the question alone. The search corpus contained 111,971 pairs of dated evidence excerpts. I combined the original search with a query-generated hypothetical passage, retained four high-ranked sources and admitted up to two outside the previous retrieved sets. This added 66 new source slots across 32 questions. Every six-source set contained the labeled current source. [Dataset, split and original comparison](../hoh_updates/README.md) · [Experiment contract](contract.json)

Qwen3.5-4B generated two question–answer cards per retrieved source before seeing the user question, one from current text and one from both editions. For each user question I also evaluated its mean token log likelihood given a current passage, a pair of different editions and a pair containing the current edition twice. To isolate the contribution of the recurrent matrices, I replaced only those matrices in the genuine pair with their no-change counterparts. Attention and convolution histories stayed on the genuine trajectory. Subtracting the two scores measured how this replacement changed support for the user question. With identical editions both contrasts were zero on all three implementation checks. [Card generation](cards.py) · [Likelihood implementation](likelihood.py) · [No-change check](null_control.json)

| Method | Exact answers / 32 | Mean token F1 |
|---|---:|---:|
| New hybrid search, read three | 3 | 0.400 |
| Read all six | 3 | 0.385 |
| Read all six with source dates | 20 | 0.790 |
| Current-passage query likelihood, read three | 3 | 0.416 |
| Current-passage query likelihood, extract from first | 25 | 0.890 |
| Ordinary span reader on six | 23 | 0.840 |
| Current-only cards and open answer | 11 | 0.575 |
| Paired-edition cards and open answer | 16 | 0.688 |
| Add whole-edition contrast | 16 | 0.678 |
| Add recurrent-matrix contrast | 15 | 0.644 |
| Copy leading current-only card | 7 | 0.348 |
| Copy leading paired card | 17 | 0.659 |

Current-passage query likelihood ranked the labeled source first on 30/32 requests. Raw whole-edition and recurrent contrasts independently ranked it first on 10 and 9 requests. Comparing two editions increased the number of questions with an exact reference answer anywhere among the six cards from 7 to 19. This measures answer availability before selecting a card. [Rank and bootstrap diagnostics](relevance.json) · [Exact scores, paired changes and model usage](summary.json)

One editorial AI reviewer examined 55 nonliteral answers with policy and rank hidden. For the ordinary top-ranked extractive reader, 29/32 answers expressed the requested fact. The reader combining cards and recurrent scores yielded 26–27/32 and the current-passage likelihood generator yielded 26–30/32. Some standalone questions omitted the film, firm or university they referred to. The intervals bracket uncertain adjudications. This post-hoc review had no independent human rater. [Rubric](semantic_scope.json) · [Scores](cold_semantic_summary.json) · [Decisions](blind_cold_assessment.json)

Promotion required three more exact answers than every conventional control and at least their highest F1. The best conventional exact score was 25/32 and the best F1 was 0.890. No proposed method passed. I left the 96 cold transfer pages sealed.

## Reproduction

The following commands rescore the archived outputs from the repository root with standard Python. The timing fields sum the experimental phases, some of which shared resources. They do not establish a serving-time speedup.

```sh
python experiments/hoh_inverse/score_inverse.py
python experiments/hoh_inverse/diagnose.py
python experiments/hoh_inverse/score_cold_semantic.py
```

For fresh inference, first rebuild the HoH bank and source rankings using the instructions in the original experiment. In an unused copy of the repository, keep the experiment directories side by side. Remove the archived append-only output files in that copy before running. Use the same pinned checkpoints and runtime versions as the original comparison. Source passages can be reconstructed from the pinned HoH dataset. [Data provenance, weights and dependencies](../hoh_updates/README.md#reproduction)

```sh
p=experiments/hoh_inverse
python "$p/query.py" "$QWEN4B"
python "$p/search.py" search "$CONTRIEVER"
python "$p/reader.py" "$ROBERTA_QA"
python "$p/cards.py" "$QWEN4B"
python "$p/search.py" cards "$CONTRIEVER"
python "$p/likelihood.py" "$QWEN4B"
python "$p/answer.py" "$QWEN4B"
python "$p/date_control.py" "$QWEN4B"
python "$p/check_null.py" "$QWEN4B"
python "$p/score_inverse.py"
python "$p/blind_cold.py"
python "$p/score_cold_semantic.py"
```

The current-passage likelihood control follows the principle of unsupervised passage reranking. Hypothetical passages and query-independent generated questions also have precedents. [UPR](https://aclanthology.org/2022.emnlp-main.249/) · [Document expansion by query prediction](https://arxiv.org/abs/1904.08375) · [Hypothetical question embeddings](https://arxiv.org/abs/2410.15262)
