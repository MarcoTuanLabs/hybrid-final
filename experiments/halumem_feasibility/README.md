# Delayed sessions in HaluMem dialogues

I used the publisher's HaluMem-Medium dialogues and official questions. I asked whether a delayed session could be identified before an evaluation that includes later questions on distinct facts. Only the first four simulated users entered this feasibility study. The first user's schema was inspected during development. Across 277 sessions and 39 official dynamic-update questions, I identified 23 earlier-event and later-question-session pairs with a second grounded question. Between each event and its later question session lay 1,630 to 12,245 tokens of intervening raw conversation (median 5,359). Benchmark annotations established eligibility. The model received only dated dialogue turns. [Selection rule](design.json) · [Episode indices](audit.json) · [HaluMem paper](https://aclanthology.org/2026.emnlp-main.220/) · [Dataset at the tested revision](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/tree/cb04336aa1b732d4b24f5186c552456b4099806e)

I selected two eligible episodes per user by a fixed hash. I removed one earlier session to model delayed arrival. I asked the original dynamic-update question, then a grounded question whose annotated evidence lay elsewhere. This second question can itself refer to another update. Both questions were revealed after the service had ingested the delayed session. The model saw its own first reply on the second turn. All three services used four retrieved dialogue turns, up to 128 tokens each, and the same four-bit Qwen3.5-9B. Reserving two results for the delayed session tests whether an explicit update notice improves on ordinary retrieval. [Service design](service_contract.json)

| Text service | Exact changed | Exact other | Semantically correct changed | Semantically correct other | Both semantically correct | Mean per episode, s |
|---|---:|---:|---:|---:|---:|---:|
| Search the old dialogue index | 0/8 | 2/8 | 0/8 | 4/8 | 0/8 | 2.45 |
| Add the session and search | 2/8 | 2/8 | 2/8 | 4/8 | 2/8 | 2.45 |
| Add the session and reserve two results for it | 1/8 | 2/8 | 1/8 | 4/8 | 1/8 | 2.49 |

The corrected dialogue index answered 2/8 changed questions and 4/8 other grounded questions under semantic grading. The old index answered 0/8 and 4/8. For 6/8 changed questions, the published reference string appeared verbatim after normalization in the actual delayed dialogue. One of the eight reference strings also appeared elsewhere in the old dialogue. This check measures literal availability and does not establish whether other wording supplies the same answer. [Grounding audit](grounding.json)

A single method-blind AI review graded 25 distinct responses against the benchmark references. These exploratory judgments and exact wording scores use sixteen questions. The eight episodes come from only four users. Times include construction and correction of the lexical index and both generated answers. Model loading and passage-independent warm-up are outside the timer. Each turn prefills the accumulated conversation. This experiment measures headroom for a subsequent native update. [Aggregate scores](text_summary.json) · [Per-episode timings, response hashes and blinded grades](observations.json) · [Grading rubric](grade_rule.txt)

## Reproduction

To verify the archived scores with standard Python, run `python experiments/halumem_feasibility/verify_archive.py`. The source dialogues and reference texts remain with the dataset publisher. The downloader retrieves the four original records unchanged from a pinned revision and verifies their byte hash. The checked-in archive contains scripts, indices, output hashes and grades. On Apple Silicon, set QWEN9B to the pinned MLX model and DATA to a local path. Run these commands from a disposable repository checkout. The scorer looks up the archived blinded judgments by a hash of the actual question, reference and prediction. A changed generation requires fresh blind grading. [Model and tokenizer hashes](../read_projection/release.json) · [Artifact provenance](bundle.json)

```sh
p=experiments/halumem_feasibility
python "$p/fetch_data.py" "$DATA"
python "$p/audit.py" "$DATA" "$QWEN9B"
python "$p/run_text.py" "$DATA" "$QWEN9B" .
python "$p/score_text.py" "$DATA"
python "$p/check_grounding.py" "$DATA"
```
