# Following edited facts through a multi-hop question

On 96 questions adapted from MQuAKE-Remastered, Qwen3.5-9B answered 63 correctly when given ranked facts. It answered 75 when a small supervised text parser decomposed each question into relations and the reader followed them in order. Adding features from a recurrent layer to the text parser also yielded 75. The 4B reader showed a similar pattern. Both sizes answered the same development questions. [Per-question scores and paired comparisons](summary.json)

| Method | 4B correct / 96 | 9B correct / 96 | 9B edited multi-hop / 24 | 9B unaffected multi-hop / 24 |
|---|---:|---:|---:|---:|
| Ranked facts | 55 | 63 | 12 | 10 |
| G-Walk prompt, surface relation matching | 61 | 66 | 16 | 9 |
| G-Walk prompt, Contriever and free-relation questions | 61 | 71 | 17 | 13 |
| Supervised lexical parser | 69 | 75 | 19 | 15 |
| Lexical and residual features | 66 | 72 | 17 | 14 |
| Lexical, residual and native token reads | 70 | 75 | 19 | 15 |
| Lexical, residual and transported writes | 68 | 73 | 20 | 12 |

Each bank contains 512 available edits. The two banks contain 48 source cases in total. Each case supplies a single-hop and a multi-hop question. Half the cases contribute an edited fact. On the 9B single-hop questions, every method recovered all 24 edited answers and 17 of 24 unaffected answers. The reader uses a shared direct-answer route for recognized single-hop questions. Multi-hop questions usually require intermediate facts from the reader's own knowledge. [Question selection, scoring and data attribution](../remastered_contrast/ATTRIBUTION.md)

The 9B lexical parser gained 17 answers and lost five against ranked retrieval. It gained six and lost two against the semantic G-Walk adaptation. The corresponding 95% paired bootstrap intervals for its accuracy advantage are 3.1 to 21.9 and −1.0 to 10.4 percentage points. Resampling groups the two questions from each source case. These intervals describe variation within this development pool. The 192 reserved cases stayed outside model selection and evaluation. [Bootstrap implementation](check.py)

## What the parser learns

A span tagger first locates the entity in the question. The parser reads the surrounding words in order of distance from that entity. A small convolutional network trained with connectionist temporal classification predicts a sequence of Wikidata relations. At each step the reader looks up the current entity and relation in the bank. If it finds an edit it uses the updated object. Otherwise it asks the language model that single-hop question. Supervision comprises 856 distinct masked question forms from 361 distinct contributing root names. The 128 validation cases select the checkpoint and decoding rule. The training, validation and evaluation construction excludes overlapping case-root IDs and gold-masked question forms. Relations come from public source annotations. The serving programs receive the inferred entity and predicted path. [Split construction](state_parser/data.py) · [Training](clause_reads/train.py)

The feature comparison uses the seventh layer of the frozen 4B model. Every supervised arm receives the same lexical and position features. Additional inputs include the residual stream, the recurrent output at each word, or the contribution of each write to the final query. Propagating the final query backward through later keys and retention gates reconstructed the projected final read with maximum batch-relative error 0.00201. Of 48 exposed multi-hop paths, lexical features recovered 39, token reads recovered 41 and transported writes recovered 40. The fixed 4B parsers supplied the paths for both final reader sizes. [Feature implementation](clause_reads/features.py) · [Original selection criterion](clause_reads/contract.json)

The G-Walk adaptations use the authors' four decomposition demonstrations with the same predicted root as the supervised methods. The semantic variant applies the original Contriever matching threshold of 0.845 to the available relation templates and generates a subquestion for unmatched relations. The adaptations use greedy decoding, one question per evaluation and the shared graph executor. Upstream G-Walk also uses case metadata and benchmark-wide entity aliases. Both published prompt files come from Shaochen Zhong and colleagues, revision 349dcc50460c251ee09a6804aab444508a016485, under CC BY 4.0. [Original implementation and prompts](https://github.com/henryzhongsc/MQuAKE-Remastered/tree/349dcc50460c251ee09a6804aab444508a016485) · [License](SOURCE_LICENSE.txt)

## A direct test of recurrent memory

A separate 4B experiment followed the old and updated paths in parallel. When the paths diverged and the updated bank lacked the next fact, the reader could adjust its recurrent matrices along the next read direction. The adjustment bounded the size of the change and removed its component parallel to the current read. The correction was eligible on 11 questions, producing 17 distinct within-question comparisons across methods. The ordinary walk, ordinary rank-one adjustment and direction-adjusted version each answered 69 of 96 correctly. Whole-prompt logit contrast answered 68. The registered accuracy criterion failed. The shared inference records support comparisons of answers. A serving-time speedup would require independent timing of each policy, including its parser and retriever. [Registered method](paired_hops/contract.json) · [Observed answers](paired_hops/4b_rows.jsonl)

## Reproduction

From the repository root, the following command verifies all archived final-answer counts and grouped intervals using the Python standard library.

```sh
python experiments/remastered_paths/check.py
```

For fresh inference, create a working directory in an unused location. The prepare phase expands the archived data and frozen parser predictions. The next commands run both readers, the paired intervention and the semantic comparator. The selected model paths should contain the corresponding local checkpoints. MLX 0.32.4 and mlx-lm 0.32.2 were used for generation. The mapping phase requires PyTorch and Transformers. The 4B reader used BF16 weights and the 9B reader used four-bit weights. [4B model](../late_arrival/model_4b.json) · [9B model](../late_arrival/model_9b.json) · [Contriever weights](https://huggingface.co/facebook/contriever-msmarco/tree/abe8c1493371369031bcb1e02acb754cf4e162fa)

```sh
script=experiments/remastered_paths/reproduce.py
run_dir=/tmp/remastered-paths-run
python "$script" prepare --out "$run_dir"
python "$script" 4b --model /path/to/Qwen3.5-4B --out "$run_dir"
python "$script" 9b --model /path/to/Qwen3.5-9B-4bit --out "$run_dir"
python "$script" paired --model /path/to/Qwen3.5-4B --out "$run_dir"
python "$script" map --model /path/to/contriever --out "$run_dir"
python "$script" semantic4b --model /path/to/Qwen3.5-4B --out "$run_dir"
python "$script" semantic9b --model /path/to/Qwen3.5-9B-4bit --out "$run_dir"
python experiments/remastered_paths/check.py --runs "$run_dir"
```

The optional `features` phase takes the 4B model and recreates the frozen feature arrays. The subsequent `train` phase refits all six sequence parsers, checks their predictions against the archive and records the refit separately. Training and final reader serving costs are distinct.
