# Selecting a delayed update before its future questions

A full dated notice matched chronological replay on the eight previously tested HaluMem development episodes. I tested whether the first three layers of the frozen model could identify which incoming turns to keep. All 23 eligible sessions from the first four development users entered the screen. Each policy retained whole dated turns in chronological order with a budget of 25% of the original session's model tokens. [Parent service comparison](../halumem_bridge/README.md) · [Frozen design](contract.json)

At the third recurrent layer I computed each turn's terminal matrix twice. The second computation set its incoming value writes to zero while holding the prior state, keys and gates fixed. I read the resulting matrices under the same four generic chat-template queries. The candidate ranked the RMS-normalized read displacement, averaged over probes and heads. A matched comparator ranked the raw displacement. Both scores were divided by the square root of turn length. Neither selector had access to a future question, reference answer or gold memory point. [Shallow selector](shallow.py)

The publisher supplied 27 changed-fact questions for these delayed sessions. For 15 questions the normalized contiguous reference phrase occurs in the actual delayed dialogue. Each row reports how often its selection retained at least one original turn containing the reference. The two paired columns count the normalized candidate's gains and losses relative to that row. This screen measures literal source coverage. Generated-answer accuracy requires a separate service experiment.

| Selection rule | Reference retained | Fraction of original tokens | Paired gains | Paired losses |
|---|---:|---:|---:|---:|
| Latest turns | 5/15 | 24.4% | 3 | 1 |
| Uniform temporal coverage | 9/15 | 24.5% | 2 | 4 |
| User turns first, recent within role | 1/15 | 24.2% | 6 | 0 |
| Within-session lexical rarity | 7/15 | 24.4% | 2 | 2 |
| Contriever session centrality | 6/15 | 24.2% | 1 | 0 |
| Raw recurrent read displacement | 8/15 | 24.2% | 0 | 1 |
| Normalized recurrent read displacement | 7/15 | 24.2% | 0 | 0 |

The prespecified condition for running full two-turn inference failed. [Selections and token budgets](selections.json) · [Reference witness indices](witnesses.json) · [Scores](summary.json)

Question-independent prompt compression has established alternatives, including LLMLingua and LLMLingua-2. The present comparison concerns isolated, head-normalized value writes as a criterion for retaining dated turns. [LLMLingua](https://arxiv.org/abs/2310.05736) · [LLMLingua-2](https://arxiv.org/abs/2403.12968)

## Reproduction

The standard-library verifier recomputes the counts and matched gains and losses. Regenerating selections requires the publisher's original development records, the pinned 9B model, and a CPU PyTorch environment for Contriever. [Dataset](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/tree/cb04336aa1b732d4b24f5186c552456b4099806e) · [Artifact hashes](bundle.json)

```sh
p=experiments/halumem_sieve
python "$p/verify.py"
python experiments/halumem_feasibility/fetch_data.py "$DATA"
python "$p/prepare.py" "$DATA" "$QWEN9B" .
python "$p/shallow.py" "$QWEN9B" .
"$TORCH_PYTHON" "$p/dense.py" "$CONTRIEVER"
python "$p/score.py"
```
