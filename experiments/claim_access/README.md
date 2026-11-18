# Finding firsthand reports in conflicting conversations

I tested whether a candidate answer drawn from a user statement is supported by the recurrent memory of that statement. I first searched dated user utterances, then used RoBERTa-SQuAD2 to propose a span. I measured the log likelihood of that span after masking its occurrence in the report. One arm replaced only the recurrent matrices with those from the masked prompt. A full-state control reread the entire masked prompt. Each selected group of statements went to the same 4B reader, which generated a free answer. Two additional controls read all statements from their respective candidate pools. [Design and gates](contract.json) · [Wide controls](wide_control.json) · [Relation to previous studies](related_work.md)

All 24 questions and eight simulated personas were previously used for development. They include eight changing facts, eight stable attributes and eight conditional preferences. The system received the raw question, date and the dialogue available at the time. Reference answers and conflict labels were withheld from inference. [Corpus and original comparisons](../memconflict_witness/README.md) · [Frozen sources](implementation.json)

| Reader | Fully correct / 24 | Mean grade | Changing / 8 | Stable / 8 | Conditional / 8 |
|---|---:|---:|---:|---:|---:|
| Previous text reader with session boundaries and retrieved evidence | 12 | 0.562 | 4 | 1 | 7 |
| Lexical and dense retrieval of user statements | 13 | 0.583 | 4 | 2 | 7 |
| Extractive answerability, top eight | 14 | 0.646 | 3 | 3 | 8 |
| Extractive answerability, chronological coverage | 11 | 0.542 | 2 | 2 | 7 |
| Answer likelihood and extractive score | 11 | 0.521 | 3 | 2 | 6 |
| Full-state phrase-masking score | 13 | 0.562 | 4 | 2 | 7 |
| Recurrent-only phrase-masking score | 12 | 0.542 | 3 | 2 | 7 |
| All statements in the causal candidate set | 13 | 0.604 | 3 | 2 | 8 |
| All statements in the extractive shortlist | 14 | 0.604 | 4 | 2 | 8 |

The strongest new conventional or whole-state comparator was Extractive answerability, top eight, with 14/24 fully correct. Recurrent-only masking reached 12/24. The preprocessing and inferential costs of the new arms differ. Bootstrap intervals apply to paired personas on this development sample, where several policies were compared.

The conventional development gate fell short. The recurrent development gate fell short.

One AI assessor graded distinct answers with policy labels hidden and retained previous grades for repeated answers. A second, related 9B model graded each distinct answer under two letter assignments. These assessments differ from the upstream judge. A static question can designate the earlier biography as true even when the simulated user subsequently supplies an explicit correction. This experiment measures agreement with that reference. [Grading protocol](assessment.json) · [Primary paired results](summary.json) · [Second-judge sensitivity](judge_sensitivity.json)

Of eight early reports identified after inspecting the static references, the previous retrieval witnesses contained two and the extractive shortlist contained four. Other reports could support the same fact. This is a reference-guided development diagnostic. The 4B reranker evaluated 308 proposed spans, and reproduced the unchanged conditional likelihood when the recurrent matrices were replaced with themselves on each question. [Witness audit](dev_witness_diagnostic.json) · [Free answers with and without the observed witness](witness_outcomes.json) · [Observed scores](supports.jsonl) · [Effect on selected evidence](ranking_diagnostic.json) · [Measured component costs](cost.json)

## Reproduction

Use standard Python for scoring. Prepare the corresponding development inputs under experiments/memconflict_witness using its pinned upstream file. Then run the commands below from the repository root. The required parent grades and all new answer strings are archived.

```sh
p=experiments/claim_access
d=experiments/memconflict_witness
python "$p/assess.py" prepare "$d"
python "$p/assess.py" score "$d"
```

For fresh inference, use a separate working copy with empty append-only outputs and regenerate the parent evidence file using the commands in the preceding experiment. The extractive and dense model file hashes are checked by the retrieval program. The offsets file records the selected locations, extracted character ranges, ranks, and scores. The full dialogue can be reconstructed from upstream. Use the archived grades only for the associated texts. [Model dependencies](../hoh_updates/README.md#reproduction)

```sh
python "$p/formulate.py" "$QWEN4B" "$d"
python "$p/retrieve.py" "$d" "$CONTRIEVER" "$ROBERTA_QA"
python "$p/support.py" "$QWEN4B"
python "$p/answer.py" "$QWEN4B" "$d"
python "$p/assess.py" prepare "$d"
python "$p/second_judge.py" "$QWEN9B" "$d"
```
