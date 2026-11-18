# Asking the same question after its source changes

This service starts with a known Wikipedia page and one question. It answers once from the earlier evidence, receives an updated passage, and answers again. The page identity is available before the first answer. If a page contains several evidence pairs, old-text query overlap determines the selection, with the date and fixed record key breaking ties. All policies see the same page and question. The page-scoped task gives the service information that the full-corpus retrieval experiment must infer. [Full-corpus study](../hoh_updates/README.md) · [Repeated-query protocol](contract.json)

On 32 development pages, Qwen3.5-4B full replay and exact prefix reuse both answered 25 questions correctly after the update. Each method also answered 25 old questions correctly. A surface-only patch with a unique-edit fallback answered 24 new questions correctly in 22.47 seconds. Recording four source-token attention weights per generated token in each earlier answer produced 24 correct new answers in 22.46 seconds. The trace-specific rule fired twice. Replaying from exact prefix checkpoints took 37.70 seconds. The reductions include the cost of the first answer and leave a one-question accuracy deficit. Direct RoBERTa-SQuAD2 on a CPU answered 28 new questions correctly in 1.19 seconds for both editions together. [All Qwen costs and decisions](summary.json) · [CPU control and timings](reader_rows.jsonl)

The text rule aligns the old and new passages. It tries to map the old answer as a contiguous span, then to substitute individual changed spans. The unique-edit policy may use a single short edited phrase when literal alignment fails. The native policy uses the model's attention over the prior evidence during answer generation to select an answer interval for replacement. Unresolved cases regenerate from a checkpoint. Each timed policy performs its own first generation. Policy order rotates by page. Trace weights are observational, and the method edits the cached answer string. [Alignment rules](repair.py) · [Model execution](run.py) · [Related work on attribution from model internals](https://aclanthology.org/2024.emnlp-main.347/)

Run `python experiments/hoh_refresh/score_refresh.py` from the repository root with Python 3.11 or later to reproduce the Qwen results. To generate new outputs, complete the preparation step in the sibling full-corpus study, then run the following in the same shell. [Data source and attribution](../hoh_updates/README.md#reproduction)

```sh
mkdir "$work/hoh_refresh"
cp experiments/hoh_refresh/*.py "$work/hoh_refresh/"
python "$work/hoh_refresh/inputs.py"
python "$work/hoh_refresh/run.py" /path/to/Qwen3.5-4B
python "$work/hoh_refresh/reader_control.py" /path/to/roberta-base-squad2
python "$work/hoh_refresh/score_refresh.py"
```
