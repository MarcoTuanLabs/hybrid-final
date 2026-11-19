# Normalization-Aware Readout for Targeted Memory Control in Hybrid Language Models

**Lu-Yen Marco Tuan**  
Department of Cognitive Science  
University of California San Diego

Four-page working paper for the AAAI 2027 Deployable AI Workshop format, November 18, 2026. [Named PDF](output/pdf/hybrid_memory_working_paper.pdf) · [Anonymous PDF](output/pdf/hybrid_dai_review.pdf) · [LaTeX](paper/main.tex)

## Results

| Experiment | Reference | Readout result |
|---|---:|---:|
| Qwen3.5-9B, independent synthetic targets | Whitened key 235/512 | Normalized reader 361/512 |
| Qwen3.5-27B, independent synthetic targets | Raw SVD 217/256 | Normalized reader 256/256 |
| Qwen3.5-27B, complete intervention time | Four starts 0.9108 s | Native start 0.7120 s |
| 4-bit Qwen3.5-9B, fresh synthetic pilot | Whitened key 15/32 | Sixteen-step reader 26/32 |

The BF16 normalized readers preserve every tested nuisance and unwritten answer. The 27B timing comparison uses the same set of 256 successful target presentations. In the small 4-bit pilot the native-start reader and the original four-start reference returned the same top token on all 128 conditions. [BF16 archived score check](scripts/check_claims.py) · [4-bit pilot](evidence/quantized_9b)

On 64 new seeds in four-bit Qwen3.5-9B (128 target presentations), the optimized read controlled 99 answers, compared with 68 under the whitened-key baseline. Giving the written heads their unwritten lengths retained 95 of the 99 successful answers. Changing only the lengths controlled 0/128. The full, direction-only and length-only readers each preserved 128/128 nuisance answers. [Head interventions, archived outcomes and seed-level intervals](experiments/read_direction/README.md)

On eight previously used 4-bit development seeds, the 16-step solver controlled 13/16 target answers, four steps controlled 3/16 and one Jacobian step controlled 0/16. Jointly prefilling both prompts saved 13.0% of complete time but changed the write in every request. Interleaving the branches at their original input shape preserved every write, query and top token across 64 requests. Its fastest tested schedule saved 5.5% relative to separate full prefills. [Solver comparison](experiments/jacobian_service/README.md) · [Prefix and batching comparison](experiments/shared_prefill/README.md) · [Layer scheduling](experiments/layer_schedule/README.md)

On 32 fresh four-bit seeds, the normalized reader controlled 57/64 targets, state-Gram whitening controlled 30/64 and four prefix-key covariance controls each scored 0/64. All seven readers preserved 192/192 controls. The comparisons use a single frozen Qwen checkpoint. [Curvature controls and the fixed-backbone comparison](experiments/covariance_read/README.md)

Predicting the layer-28 donor write from layer 12 reduced complete latency by 24.9% against exact truncation, but controlled only 3/16 targets. Exact truncation controlled 13/16 and two-row batching controlled 14/16. [Pilot with calibration and timings](experiments/shallow_donor/README.md)

## Country–capital bindings before and after a memory edit

On the public CAPITALS templates from Feng and Steinhardt, the fixed answer-time two-head reader controlled 0/24 changed answers. With the update applied *before* any of the three questions, the donor's attention keys and values at the two country tokens and their following punctuation controlled 40/48 changed capitals. Full replay controlled 40/48, with two paired gains and two losses. Replacing all recurrent and convolution states alone controlled 0/48. Predicting the same four local K/V replacements from the old document controlled 35/48 on the development prompts. [Fixed-reader transfer](experiments/binding_transfer/README.md) · [Which cache entries carried the change](experiments/binding_cache/README.md) · [Donor-free estimation](experiments/binding_transport/README.md)

On the 64 prospectively registered documents, the four-position donor attention transplant controlled 98/128 revised capitals. Full replay controlled 105/128, transfer of all recurrent and convolution states controlled 0/128, and the donor-free query redirect controlled 99/128. [Independent CAPITALS confirmation](experiments/binding_confirmation/README.md)

## Public cache-transfer experiment

The receiver starts with an old, question-conditioned cache and also receives the donor's attention cache and convolution histories. The encoder projects the donor-minus-recipient recurrent matrix onto the directions read during four known suffix tokens. It sends 3 MiB of recurrent-matrix updates, compared with 48 MiB for an FP32 dense transfer. These sizes exclude the common donor components. The receiver reconstructs a dense state locally.

On 32 official bAbI test stories and 32 previously unused MQuAKE cases, the projected update reproduced all 128 dense answer sequences. Equal-size rank-four SVD reproduced 124. Dense transfer and the projection answered 53/64 updated questions correctly. SVD answered 54/64. The preregistered confirmation rule also required halving centered-logit error against each compressed control on both tasks. The strongest comparison gave error ratios of 0.508 on bAbI and 0.378 on MQuAKE. The composite rule therefore failed on bAbI. No system latency, general factual accuracy, or resident-memory advantage is asserted. [Experiment and rerun instructions](experiments/read_projection) · [Held-out scores](evidence/read_projection/test_summary.json) · [Other public-task pilots](evidence/public_diagnostics/README.md)

On 16 short public development requests, sending a rank-four recurrent update along with changed attention and convolution state required 6.78 MiB per request on average. The complete projected service took 0.843 seconds per request, compared with 0.551 for utf-8 splice and checkpoint reuse. Both figures include generation and actual local socket transfer. [Complete update costs](experiments/update_service/README.md)

I also fitted a shared read basis once, then updated a document before either downstream question was known. On 12 separate development documents it reproduced 19/24 dense answer sequences. Its mean warm time including both future questions was 0.552 seconds versus 0.669 for text splice with a prefix checkpoint over a local 1-Gbit/s emulated link. [Shared-document service and all controls](experiments/shared_read_basis/README.md)

## Updating a cached conversation before the question

On 34 held-out questions from shortened LongMemEval histories, a per-head correction reproduced the 4-bit 9B full-replay answers apart from one final period. Each method matched 21 reference answers. Counting the original history, insertion and answer, estimated decoder arithmetic fell by 25.3% and measured time by 19.5%. Reusing the old recurrent state also scored 21/34 at nearly the same cost. BM25 scored 19/34 with the lowest computational cost. [Methods, competing results and reproduction](experiments/late_arrival)

On the 16 shortened 4B development histories, the correction reproduced all 16 full-replay sequences and reduced total elapsed time from 99.36 to 78.38 seconds. Appending each delayed session with its original date was faster still and answered 13/16 correctly, compared with 12/16 for chronological replay and correction. [Append-only comparison](experiments/late_arrival/README.md#appending-a-delayed-dated-session)

A separate balanced MQuAKE-Remastered pilot compared six readers on edited and unaffected single- and multi-hop questions. The conventional scoped reader scored 55/96 and the recurrent-channel contrast scored 54/96. [Per-task results and controls](experiments/remastered_contrast)

On 12 development documents with three-turn conversations, the old cache with a change notice and retrieval answered the changed question correctly in both phrasings on all 12 documents. Adding a rank-four recurrent patch changed no answer's correctness and raised mean cold service time from 1.704 to 1.963 seconds. [Pre-question updates across three turns](experiments/persistent_notice/README.md)

In HaluMem-Medium, four simulated users yielded 23 delayed-session episodes with a second grounded question. An eight-episode text-only pilot measured how many official answers lexical retrieval could recover after the delayed session arrived. [Delayed dialogue eligibility and measured text controls](experiments/halumem_feasibility/README.md)


On all 23 eligible development sessions, keeping one quarter of each incoming session by normalized shallow-state read change preserved the reference phrase for 7/15 attainable dynamic-update questions. Uniform temporal coverage preserved 9/15, and raw read displacement preserved 8/15. The normalized selector failed the criterion for a generation test. [Quarter-budget update selection](experiments/halumem_sieve/README.md)

On eight delayed HaluMem development sessions, I charged the original ingest, cache update, neural retrieval and two generated answers. The anchor correction answered 5/8 changed and 4/8 other grounded questions. The matched partial replay answered 5/8 and 4/8. [Chronological cache intervention and matched controls](experiments/halumem_bridge/README.md)

## Following edits across several facts

On 96 corrected MQuAKE development questions, a supervised text parser followed by Qwen3.5-9B answered 75 correctly. Ranked retrieval answered 63 and an adaptation of G-Walk with semantic relation matching answered 71. Adding recurrent token reads to the text parser also yielded 75. The 4B comparison yielded 69, 55, 61 and 70, respectively. Both readers used the same questions and graph. [Method, archived answers and reproduction](experiments/remastered_paths/README.md)

## Dated Wikipedia revisions

On 224 independent VitaminC test pages, a paired readout of ALBERT-xlarge scored 373/448 endpoints, the original specialist scored 365/448, and the strongest simpler comparison scored 372/448. [Registered specialist transfer and every comparison](experiments/vitaminc_transfer)

On 44 new VitaminC development pages, a reader that jointly predicted the two revised verdicts from Qwen's middle layer scored 65/88, full Qwen replay scored 52/88, and published ALBERT-xlarge scored 67/88. [Earlier paired pilots](experiments/vitaminc_read) · [Stratified natural revision pilot and all comparators](experiments/vitaminc_joint)

On 20 DynaQuest pages with changing answers, Qwen3.5-4B answered 28 of 40 dated questions correctly when shown both editions and 25 when shown only the requested edition. A factorial cache probe compared the old and new reference answers directly. On the six pages with two errors in the 4B mixed-context baseline, 9B favored the correct pairing on five but freely produced both answers on zero. The attention cache supplied the larger median contribution to the old/new log-odds difference at both model sizes. [Dated evidence, scope and complete results](experiments/dated_versions/README.md)

## Answering the same question after a page changes

On 96 held-out HoH-QAs pages, a text-based update to a saved answer reduced measured Qwen3.5-9B time by 33.9% and estimated arithmetic by 28.2% relative to exact prefix reuse and regeneration. The methods matched 80 and 79 current reference answers, respectively. The faster method recovered two answers and lost one. A CPU span reader using the same rule answered 77/96 in 2.07 seconds, compared with 75/96 in 3.66 seconds for full rereading. Timing includes both editions. A post-hoc blinded AI review found that 96/96 rereads and 95–96/96 repaired answers conveyed the requested reference fact. [Methods, raw outputs and paired intervals](experiments/hoh_boundary/README.md) · [Semantic review](experiments/hoh_semantic/README.md)

In a separate 32-question full-bank experiment, the best direct readers answered 23 questions correctly. Neither attention-cache nor recurrent-matrix contrast improved the 22/32 score of the matched expanded-retrieval generator. [Full-bank evaluations](experiments/hoh_updates/README.md)

On the same 32 cold questions, a current-passage likelihood ranker followed by a span reader answered 25/32 exactly. It ranked the labeled source first on 30/32 requests. Adding a recurrent-matrix contrast to cross-edition cards yielded 15/32 with free generation. [Cold retrieval and answer comparison](experiments/hoh_inverse/README.md)

On 32 human-written DiffQG questions adapted for open QA, undoing one edit before scoring an answer span yielded 23 exact answers. A simpler free reader with changed-span suggestions also scored 23. [Independent human-question pilot](experiments/diffqg_hunks/README.md)

On 24 MemConflict requests drawn from eight simulated personas, a speaker-label contrast restricted to the recurrent matrices recovered 10 fully correct answers. A conventional text reader combining session-boundary statements with retrieved witnesses recovered 12. A policy-blind AI reviewer applied the benchmark's semantic criteria. [Conversation methods, prompts and all graded results](experiments/memconflict_witness/README.md)

On the same 24 requests, a follow-up isolated the effect of masking a proposed answer in a firsthand statement. Recurrent-only masking recovered 12/24 fully correct answers. The strongest new comparator, extractive top-eight retrieval, recovered 14/24. [Firsthand-statement retrieval and masking](experiments/claim_access/README.md)

## Build and verify

```sh
bash scripts/build.sh
python scripts/check_claims.py
python experiments/quantized_write/score.py
python experiments/read_projection/score.py dev
python experiments/read_projection/score.py test
```

The build uses the bundled AAAI 2027 style and PDFLaTeX. Set `PDFLATEX` to an executable path for a nondefault TeX installation. The published figure can be regenerated with `python scripts/plot_results.py`. Model-free scoring uses NumPy. The fresh inference programs, model revision, split inputs, labels and per-question outputs are included. [Release check](qa/review.txt) · [Electronic file manifest](qa/release_manifest.json)

The earlier synthetic results and raw scores were recovered from the original complete archives. [Source snapshot](https://github.com/MarcoTuanLabs/hybrid-answer-relay/tree/8dd159012ec7cccf3802bafa6f436b7ea0f09e61) · [Archived result provenance](evidence/provenance.json) · [Previous working paper](https://github.com/MarcoTuanLabs/hybrid-final/tree/42d5e72a1832f0f6efe1fbbbb546ed3a55a807ce)

The manuscript, implementation and release checks were prepared with AI assistance for the named author's review.
