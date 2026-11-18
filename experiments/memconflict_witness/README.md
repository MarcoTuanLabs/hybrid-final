# Reading conflicting statements in a conversation

I selected eight synthetic personas from MemConflict and asked one question per persona about each of three situations. Dynamic questions ask what changed. Static questions test the stable fact designated by the benchmark when later statements conflict with it. Conditional questions ask when a preference applies. The system saw the question and dialogue through the query session. It did not receive the profile, event outlines, conflict category or later dialogue. Sixteen further personas were reserved. [MemConflict paper](https://arxiv.org/abs/2605.20926) · [Upstream release](https://github.com/TaoZhen1110/MemConflict/tree/ec51d5d36e87f7665d1337f3a88cbde95fc2a964) · [Selection and hashes](selection.json)

I combined lexical and dense retrieval of dated dialogue excerpts. I compared the first six hits with a list that favored distinct sessions, and with a list that included early, late and current-session evidence. Each excerpt preserved the speaker labels. I then asked whether the recurrent matrices helped the reader distinguish what the user said from what the assistant suggested. At the first eight decoding steps, I contrasted scores under the true speaker labels with scores under swapped labels. The whole-cache control swapped the complete initial cache. The recurrent arm swapped only the recurrent matrices. Both arms shared the actual evidence and generated prefix. An additional conventional control read the first user message from each prior session and all user messages from the current one. It used the benchmark convention for stable attributes. [Registered comparisons](contract.json) · [Additional control and its timing](boundary_control.json)

| Reader | Fully correct / 24 | Mean graded score | Dynamic / 8 | Static / 8 | Conditional / 8 |
|---|---:|---:|---:|---:|---:|
| Six ranked excerpts | 10 | 0.438 | 2 | 2 | 6 |
| Six, favor distinct sessions | 9 | 0.417 | 2 | 1 | 6 |
| Up to 12 diverse excerpts | 10 | 0.438 | 3 | 2 | 5 |
| Wide, conflict-aware prompt | 11 | 0.500 | 2 | 2 | 7 |
| Early, late and current witnesses | 9 | 0.417 | 2 | 1 | 6 |
| Witnesses, conflict-aware prompt | 10 | 0.438 | 3 | 1 | 6 |
| Witnesses and a second-pass critique | 11 | 0.479 | 3 | 1 | 7 |
| Session-boundary statements | 2 | 0.188 | 1 | 1 | 0 |
| Boundary statements and witnesses | 12 | 0.562 | 4 | 1 | 7 |
| Whole-cache speaker contrast | 10 | 0.458 | 3 | 0 | 7 |
| Last completed session | 3 | 0.208 | 3 | 0 | 0 |
| Recurrent-matrix speaker contrast | 10 | 0.438 | 3 | 1 | 6 |

One AI reviewer graded 178 distinct case/answer pairs with method labels and timing hidden. Dynamic and static answers received 0, 0.5 or 1. Conditional answers received 0 or 1. The primary comparison uses the count receiving 1. The original benchmark uses a different language-model judge. Some static reference facts conflict with a later explicit correction by the simulated user. Performance here measures agreement with the benchmark’s designated answer. [Per-persona paired differences and gates](summary.json) · [Assessment notes](assessment_notes.json) · [Blind grades](grades.json) · [Official grading rules](https://github.com/TaoZhen1110/MemConflict/blob/ec51d5d36e87f7665d1337f3a88cbde95fc2a964/Evaluation/eval_scoring.py)

No proposed arm gained the required three fully correct answers over the strongest applicable controls with a preserved mean score. I left the transfer personas sealed.

## Reproduction

Download the original evaluation file and use an unused working copy. The builder verifies the source digest, reconstructs the selected questions and references, and writes dialogue histories locally. The release stores retrieval locations, predictions and decisions. Model-free rescoring requires standard Python.

```sh
p=experiments/memconflict_witness
python "$p/prepare.py" /path/to/Step4_4.jsonl
python "$p/adjudicate.py" prepare
python "$p/adjudicate.py" score
```

For fresh answers, use an unused copy without the archived append-only prediction files. Run the following after data preparation. Blinded grades apply to the archived texts. New wording requires fresh adjudication. [Pinned models and environments](../hoh_updates/README.md#reproduction) · [Dialogue annotations](https://github.com/TaoZhen1110/MemConflict/blob/ec51d5d36e87f7665d1337f3a88cbde95fc2a964/Data/Step4_4.jsonl)

```sh
python "$p/expand.py" "$QWEN4B"
python "$p/retrieve.py" "$CONTRIEVER"
python "$p/run.py" "$QWEN4B"
python "$p/boundary_control.py" "$QWEN4B"
python "$p/latest_control.py" "$QWEN4B"
```
