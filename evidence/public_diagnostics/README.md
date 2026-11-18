# Transferring the readout to a live quantized model

The headline BF16 experiments use independent synthetic retrieval cohorts. A new Apple M5 pilot used a four-bit MLX conversion of Qwen3.5-9B, revision 8b2b98c00a6b4d291155e4890773ca8f769aee53. Sixteen new seeds each contributed a table and prose presentation. Six readers used the same two-head, 256-coordinate value write at layer 28 and the same four available suffix positions. The 768 actual continuations include target writes, unrelated writes, and both corresponding unwritten controls.

| Reader | Targets controlled / 32 | Unaffected answers preserved / 96 |
|---|---:|---:|
| Native query | 0 | 96 |
| Whitened key | 15 | 96 |
| Six ordinary gradient steps | 13 | 96 |
| Six projected conjugate-gradient steps | 20 | 96 |
| Sixteen gradient steps from the native query | 26 | 96 |
| Eighty gradient steps from each of four starts | 26 | 96 |

The two longer solvers produced the same top token on every paired request. Relative to whitened key, the sixteen-step reader gained eleven targets and lost none. The descriptive 95% paired seed-bootstrap interval for its 34.4 percentage point advantage spans 15.6 to 53.1 points. This small pilot adds a quantized-model check. It does not establish equivalence of the two solvers on unseen inputs.

The six-step solver transferred poorly despite retaining almost all of the reference objective on 768 public-task query positions on 4B. Its 4B median retained gain was effectively 100% and the tenth percentile was 99.6%. On the larger model the six-step solver lost six targets relative to the sixteen-step reference. A numerical optimizer score on 4B could not substitute for behavior on 9B.

## Cost of fitting the query inside the target layer

A subsequent experiment on eight already exposed 9B seeds used six implementations and 384 independently timed full requests. Within a single edited layer, lower layers receive the same suffix tokens regardless of that layer's read. The reader can therefore compute its four queries at that layer while processing the suffix as a single batch. A staged control used the same batch shape and solver and truncated its planning pass at the target layer.

Fusing planning and execution preserved all 64 top tokens and all 256 chosen queries relative to the staged batched reader. The two readers controlled 13/16 targets and preserved 48/48 control answers. Average complete service time fell from 456.9 to 418.8 ms, an 8.3% reduction with a paired seed-bootstrap interval of 7.6–9.3%. This fell below the frozen 10% threshold. An equally batched implementation that observed both local recurrent states took 418.3 ms and produced the same answers. The serial inline reader controlled 14/16. The 64 fresh seeds reserved for a passing candidate remain unopened.

## What limits transfer to public questions

On the 24 exposed MQuAKE-CF-3k-v2 multi-hop cases, the four-bit 9B model answered 20/24 correctly when given the updated facts and 0/24 from the old facts. A 256-coordinate donor write controlled 0/24 complete new answers under all four tested readers, including the normalized reader applied after each generated token. All seven methods produced freely decoded answers for the same 24 target and 24 unrelated questions, for 336 outputs in total. The updated prompt answered 10/24 unrelated probes, compared with 9/24 for each write method. This test used an existing, question-conditioned donor and a fixed head pair selected for letter retrieval.

For bAbI QA1, an independent public synthetic story corpus, I selected 24 training stories. Each permitted a pair of authentic prefixes with different official answers to the same question and an unaffected person in both prefixes. A complete updated prefix answered 22/24 changed and 23/24 stable questions correctly. All six constrained transfer methods returned 0/24 changed and 24/24 stable correct answers. Supplying the full donor query or orienting the optimizer toward the full donor's normalized read did not recover a target. The 384 generated answers use ordinary location names and full-string scoring. The official bAbI test set remains unopened. [bAbI source and license](https://huggingface.co/datasets/facebook/babi_qa) [MQuAKE source](https://github.com/princeton-nlp/MQuAKE)

TimeQA supplied a separate check on human-written questions. On 32 previously exposed date questions in 16 articles, the strongest preceding short-passage reader answered 18. A four-way year and memory interaction scored 11, plain verification scored 12, and only 22 candidate pools contained a correct answer. I discarded the verifier.

## Reducing the rank of the solver computation

On 256 archived 9B slots a small-Gram initializer averaged 0.139 ms, compared with 0.155 ms for a thin SVD and 4.887 ms for dense Cholesky and SVD. All three and the original implementation differed from the archived rendering in the same single BF16 coordinate out of 98,304 after matching source scalar types. Normalized directions agreed, but one coordinate lay at a rounding boundary for the cross-backend query norm.

Replacing the second dense product throughout the four-start solver was slower. Full design took 3.01 ms per slot with the factorized operator and 2.34 ms with the same initializer and dense operator. Three of 32,768 final BF16 coordinates differed. The maximum relative objective difference reached 1.70e-5. Both the quality and speed criteria failed. Factorizing a small write reduces the cost of finding start directions but does not justify the full solver under this implementation.

The new measurements used local compute. New GPUtw charges and Adaption credits used were both zero.

## Which bank carries the public answers?

I transplanted complete cache banks on the same two 24-case development cohorts. Every arm ran the four template tokens in serial and decoded a new answer from the full vocabulary. The recipient and donor contain different prefixes. The attention-only comparison replaces all eight attention caches together, so their rotary offsets remain consistent. The recurrent-only arm replaces all 24 recurrent layers, including their convolution history and their matrix state.

| Donor material transplanted | bAbI updated / 24 | bAbI stable / 24 | MQuAKE updated / 24 | MQuAKE unrelated / 24 |
|---|---:|---:|---:|---:|
| None | 0 | 24 | 0 | 9 |
| All attention caches | 1 | 24 | 12 | 10 |
| All recurrent layers | 5 | 24 | 5 | 8 |
| Upper-half recurrent layers | 5 | 24 | 5 | 9 |
| Only recurrent matrices | 0 | 24 | 2 | 9 |
| Only convolution histories | 0 | 24 | 0 | 9 |
| Both complete banks | 22 | 23 | 20 | 10 |

Neither bank alone recovered 75% of the full donor's correct updates on both tasks. The complete donor has unequal access to information across the two single-bank arms, which have different payloads. These results support testing joint cache transfer. Prior work already introduced the bank-swap diagnostic and reported attention-dominated retrieval on its tasks. [Afendulev et al.](https://aclanthology.org/2026.findings-emnlp.2061/)

## A smaller update reproduces the dense donor

I encoded only the part of each initial recurrent-matrix difference read by the donor's four known suffix positions. The keys and update gates transport each suffix query back to the boundary of the prefix. The sender transmits two FP32 rank-four factors for each of 768 value heads, for 3 MiB rather than the dense 48 MiB. The recipient still receives the donor attention caches and convolution histories and rebuilds dense recurrent matrices locally. This is a payload comparison for a question-conditioned cache.

On the 96 development requests, the method reproduced all 96 dense sequences and passed every frozen development gate. I then opened the official bAbI test source and separately selected untouched MQuAKE cases under a precommitted rule. Each of 32 cases per dataset contributed two requests, for 128 held-out sequences. The projected update reproduced 128, rank-four SVD reproduced 124, the raw query basis reproduced 126, and 3.375 MiB rowwise two-bit quantization reproduced 123. Dense and projected transfer each answered 53 of 64 changed questions correctly, compared with 54 for SVD. The observed benefit is higher fidelity to the donor at the payload of the matched SVD comparator.

The independent gate also required halving mean centered-logit RMSE against every compressed control in each dataset. The ratio to the strongest comparator passed on MQuAKE at 0.3780 and missed on bAbI at 0.5075. The composite gate failed. The literal sequence counts stand. They do not establish a factual accuracy gain, a guaranteed twofold error reduction, a reduction of total resident memory, or faster service.
