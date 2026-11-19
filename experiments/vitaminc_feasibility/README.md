# Naturally revised evidence with publisher-labeled controls

I counted eligible revisions in the development split of VitaminC, a public fact-verification benchmark based on Wikipedia revisions and synthetic edits. This inventory uses only examples marked real by the publisher. A changed or stable claim must have an unambiguous label for each of the two exact evidence texts, on the same page and revision. The grouping treats the two evidence versions symmetrically. It makes no assumption about their order in the release. [Schuster, Fisch and Barzilay, NAACL 2021](https://aclanthology.org/2021.naacl-main.52/) · [Pinned dataset](https://huggingface.co/datasets/tals/vitaminc/tree/be6febb761b0b2807687e61e0b5282e459df2fa0)

Of 63,054 development rows, 42,290 are marked real. They yielded 11,839 natural revision-and-evidence pairs with at least one claim labeled under both texts. In 46 pairs across 21 pages, at least one claim changes label and at least one keeps its label. A separate screen found 3,407 pairs across 1,252 pages with at least two changed claims and one contiguous replacement of one to eight whitespace-delimited words on each side. Only six of those compact replacements had a labeled stable claim in the same evidence pair. These are eligibility counts. No model answers or official test-split records entered the inventory. [Counts and eligible-pair hashes](summary.json)

Download the publisher's pinned development JSONL and run the script with standard Python. The script checks its SHA-256 before reading it.

```sh
python experiments/vitaminc_feasibility/inventory.py "$VITAMINC_DEV_JSONL"
```
