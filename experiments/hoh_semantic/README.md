# Comparing the facts conveyed by repaired and regenerated answers

Normalized exact match on the 96 held-out page subscriptions favored Qwen3.5-9B answer repair by one question, 80/96 versus 79/96. Many answers that miss the reference string give the requested fact. I therefore added a post-hoc check of factual agreement with the current reference, using the question and the supplied passage. One AI editorial assessor reviewed the distinct nonliteral responses with reader, policy, timing and other responses hidden. This is a sensitivity analysis by the authoring assistant. It has no independent human ratings. [Frozen rubric](protocol.json) · [Primary registered scores](../hoh_boundary/test_summary.json)

| Model | Full model answer | Answer repair | Paired difference |
|---|---:|---:|---:|
| Qwen3.5-9B with shared prefix | 96/96 | 95–96/96 | −1 to 0 |
| Qwen3.5-9B without shared prefix | 96/96 | 95–96/96 | −1 to 0 |
| RoBERTa CPU span reader | 89/96 | 90/96 | +1 |

There were 139 distinct question–response pairs across the six policies. Eighty-one literally matched the reference after case and whitespace normalization. The assessor reviewed all 58 others, accepting equivalent numbers, names and paraphrases and requiring all explicitly requested components. One response repeated a person's surname. Interpreting the repetition as a stutter or an unsupported full name gives the two endpoints above. For example, the reference “6” and the generated answer “Six volumes.” identify the same count. In a two-part question about a polo player's handicaps, “3-goal” omits the requested arena handicap.

The review supports the timing result under a range of answer wordings. Its 9B judgments leave open the loss of one factually adequate response. They supply no evidence for a factual accuracy gain over full 9B regeneration. [Blinded cases and Wikipedia attribution links](blind.json) · [Individual judgments and reasons](adjudication.json) · [Policy totals and paired bounds](summary.json)

To recompute the table, run `python experiments/hoh_semantic/score.py`. To rebuild the blinded cases, first regenerate `test_inputs.json` using the published page-subscription preparation, copy the archived CPU, Qwen and no-checkpoint prediction files to that working directory, and run `blind.py` from a sibling `hoh_semantic` directory. The original Wikipedia evidence is distributed under CC BY-SA 4.0. [Dataset provenance and data terms](../hoh_updates/README.md#reproduction) · [Creative Commons license](https://creativecommons.org/licenses/by-sa/4.0/)
