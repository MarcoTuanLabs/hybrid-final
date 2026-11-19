# Repairing a chronological cache after a delayed HaluMem session

**Correction, November 18, 2026.** The original anchor-correction run passed negative offsets to the MLX rotary kernel when removing position from a key. A round-trip test on the pinned model showed that this changed the key and could change an answer without any edit. The bridge row, its 13.3% time saving and the conclusions about the proposed correction are withdrawn pending a complete rerun of all six services. The other five methods do not use this inverse. The original observations below remain available for provenance. The paper's 9B/27B readout and solver results use separate code.

I inserted an entire dated dialogue session into a previously ingested history before the model saw either of two future questions. The first official HaluMem question tests a changed fact. The second is a grounded question about other evidence. The second turn includes the first generated answer. All methods use the same current evidence from a BM25 shortlist reranked with Contriever. The eight development episodes come from four simulated users. [Case selection](../halumem_feasibility/README.md) · [Frozen experiment](contract.json)

The service retained a 2,048-token window of unchanged later dialogue. Partial replay revisited the first 32 and last 48 tokens, reused 1,968 intervening tokens and repositioned their attention keys. The candidate also estimated a change in the intervening keys and values from the 32 matching tokens. Competing policies use exactly the same received raw session. In particular the full notice appends every original turn in that session to the old cache. [Method and notation](method.md) · [Related work](related_work.md)

| Service | Changed fact | Other grounded fact | Both correct | Identical sequences vs. exact replay | Mean complete time, s | Estimated operations per episode, trillions* |
|---|---:|---:|---:|---:|---:|---:|
| Fresh model + updated retrieval | 2/8 | 4/8 | 2/8 | 4/16 | 7.06 | 7.06 |
| Old cache + updated retrieval | 2/8 | 4/8 | 2/8 | 4/16 | 12.33 | 22.34 |
| Old cache + full dated notice + retrieval | 5/8 | 4/8 | 2/8 | 7/16 | 20.16 | 44.24 |
| Exact chronological replay from checkpoint | 5/8 | 4/8 | 2/8 | 16/16 | 24.19 | 58.62 |
| Partial replay | 5/8 | 4/8 | 2/8 | 5/16 | 20.69 | 44.28 |
| Partial replay + anchor correction | 5/8 | 4/8 | 2/8 | 6/16 | 20.97 | 44.28 |

The candidate saved 13.3% of full elapsed time relative to exact chronological replay. Its time ratio was 0.867 (95% paired bootstrap interval 0.797 to 0.935, resampling four users). Relative to the matched partial replay, the anchor correction gained 0 changed answers and lost 0. It gained 0 other grounded answers and lost 0. The preregistered accuracy condition failed. The preregistered efficiency-and-preservation condition failed. [Paired comparisons and stage timings](summary.json)

*The computation estimate sums dense-equivalent multiply-accumulates for the decoder and retrieval encoder and adds a conservative scalar allowance for the bridge correction. It is an estimate of arithmetic, not a hardware performance counter. The complete wall clock includes original indexing and ingest, the cache update, online lexical and neural retrieval, and two generated answers. Model loading and a warm-up precede timing. Original logical checkpoint sizes and numbers of model tokens are archived per episode. All arms run on one host with an Apple M5 GPU and a CPU retrieval worker. The evidence pool is the full available transcript. The active decoder window is shorter. Four users give imprecise intervals.

A single method-blind AI reading compared unique responses with the publishers' references. It scored 53 new responses and reused 3 identical, previously graded responses. Generation stopped at the first newline or at 48 tokens. Of the 96 generated turns, 19 reached the length cap and nine responses ended at a colon. Some multi-part answers therefore received no credit before the model could elaborate. These adapted development scores measure this serving task. The public archive includes source indices, selected retrieval offsets, exact-match and semantic scores, output token hashes and measured times. Original dialogues and reference text are obtained from the publisher. [Observed results](observations.json) · [Grading rule](../halumem_feasibility/grade_rule.txt) · [Artifact hashes](bundle.json) · [HaluMem source](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/tree/cb04336aa1b732d4b24f5186c552456b4099806e)

## Reproduction

Standard Python verifies the archive from the repository root. Fresh inference requires MLX, the pinned four-bit 9B decoder and a separate CPU environment with PyTorch and Transformers for the pinned Contriever weights. In a disposable copy, obtain the four unmodified development records using the original downloader, then run the commands below. The scorer checks all cases and same-query retrieved passages. Identical question, reference and response triples reuse their archived grades. Other replies need a new blinded evaluation. [Model pin](../read_projection/release.json)

```sh
p=experiments/halumem_bridge
python "$p/verify.py"
python experiments/halumem_feasibility/fetch_data.py "$DATA"
python "$p/experiment.py" "$DATA" "$QWEN9B" . "$CONTRIEVER" --reranker-python "$TORCH_PYTHON"
python "$p/score.py" "$DATA" .
```
