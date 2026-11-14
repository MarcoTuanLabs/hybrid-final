# Normalization-Aware Readout for Targeted Memory Control in Hybrid Language Models

**Lu-Yen Marco Tuan**  
Department of Cognitive Science  
University of California San Diego

Working paper, November 14, 2026. **Four pages total, including references.**

[Read the paper](output/pdf/hybrid_memory_working_paper.pdf) · [LaTeX source](paper/main.tex) · [Pre-push review](qa/review.txt)

This paper presents the positive controlled synthetic-retrieval results from [hybrid-answer-relay](https://github.com/MarcoTuanLabs/hybrid-answer-relay), focusing on normalization-aware readout and the final solver-efficiency confirmation.

| Independent confirmation | Baseline | Normalized reader | Paired gain |
| --- | --- | --- | --- |
| Qwen3.5-9B: 256 seeds, 512 target presentations | Whitened key: 235/512 (45.9%) | 361/512 (70.5%) | +24.61 percentage points; 95% CI [20.51, 28.71] |
| Qwen3.5-27B: 128 seeds, 256 target presentations | Raw SVD: 217/256 (84.8%) | 256/256 (100%) | +15.23 percentage points; adjusted 97.5% CI [9.38, 21.88] |

The normalized reader preserves every tested answer in all three nuisance/unwritten control arms: 512/512 per arm at 9B and 256/256 per arm at 27B. The native-start 27B solver reduces mean total intervention time from 0.9108 to 0.7120 seconds, **21.83%**, relative to the four-start, 16-step reference, with 256/256 target successes for both.

These are archived synthetic-retrieval findings in two sizes of one model family. The manuscript is a focused working paper, not a new model experiment, conference acceptance, or claim of general natural-document editing performance. The two models use different fixed write paths. All within-model comparisons share the write and available query slots.

## Build

With a PDFLaTeX distribution containing the standard packages used by the included AAAI 2027 preprint style:

```sh
bash scripts/build.sh
```

To select a particular TeX installation, set `PDFLATEX` to its executable path. The included style is used in `preprint` mode, with named authorship and no conference-publication copyright claim.

## Recompute the claims and figure

Using Python with NumPy and Matplotlib:

```sh
python scripts/check_claims.py
python scripts/plot_results.py
bash scripts/build.sh
```

The independent checker reads **all 28,672 score records** from both headline cohorts. It verifies the complete experimental grids, target counts, format-specific counts, all preservation arms, paired seed-bootstrap intervals, timing reduction, reference-answer preservation, and shared-reader parity. No GPU or model weights are required. This reproduces archived-score analysis; it does not rerun inference.

## Evidence and provenance

- [Original 9B aggregate analysis](evidence/9b_results.json) and [27B aggregate analysis](evidence/27b_results.json).
- Complete archived score rows, losslessly compressed: [9B](evidence/9b_rows.jsonl.gz), [27B](evidence/27b_rows.jsonl.gz). All reader arms and outcomes remain in these files.
- [Provenance and hashes](evidence/provenance.json), original execution receipts, [claim-check result](qa/claims_check.json), and [PDF-check result](qa/pdf_check.json).
- Source snapshot: [`8dd1590`](https://github.com/MarcoTuanLabs/hybrid-answer-relay/tree/8dd159012ec7cccf3802bafa6f436b7ea0f09e61), branch `codex/native-natural-maintrack-v1`. The original long-form manuscript is [here](https://github.com/MarcoTuanLabs/hybrid-answer-relay/blob/8dd159012ec7cccf3802bafa6f436b7ea0f09e61/interpretability/paper/native_predictive_reading_review_v4.tex).

Raw score rows were recovered from the original complete archives. Both archive SHA-256 values match the published source manifests; packaged rows match the archive members and execution-receipt hashes. The aggregate result Git blob IDs match the source snapshot. Full model tensors, queries, and inference implementation remain in the source repository's archive packages rather than being duplicated here.

The manuscript and checks were prepared with AI assistance for the named author's review. The pre-push check is an assistant audit, not independent human peer review.
