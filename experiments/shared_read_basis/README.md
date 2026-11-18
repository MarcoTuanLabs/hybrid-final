# Updating a shared document cache before the questions arrive

I fitted one rank-four basis per recurrent value head on twelve development documents. For each calibration document, the updated model read the first four fixed chat-template boundary tokens. I transported those queries to the document boundary and fitted the basis to their directions. At service time a sender projects the new recurrent-matrix difference onto the installed basis, without an online suffix pass. Only four FP16 coefficients per value dimension are transmitted. The sender also supplies the updated convolution histories and changed portion of attention KV. The receiver reconstructs a dense cache before it sees the future question. [Frozen design](contract.json) · [Previous designs](related_work.md)

I evaluated separate development documents, six bAbI stories and six MQuAKE fact lists. Each had one changed and one stable question. The sender operated on the facts alone. Both questions were answered independently by the receiver after one cache update. These documents were available to earlier project development. Strong controls use full or prefix-reused replay of an exact text patch, two levels of row quantization, old matrices, and two equally sized alternative bases. All policies ran in randomized order on the same M5 GPU. Actual bytes crossed a local socket throttled to 1 Gbit/s. The throttle emulates a link. Model loading was outside the timer.

| Policy | Exact dense answers / 24 | Reference answers correct / 24 | Mean wire MiB | Warm update-ready ms | Warm update + both questions, s | Cold update + both questions, s |
|---|---:|---:|---:|---:|---:|---:|
| Text splice and fresh prefill | 23 | 18 | 0.0001 | 190.2 | 0.6982 | 0.6982 |
| Text splice with a prefix checkpoint | 24 | 18 | 0.0001 | 154.1 | 0.6694 | 0.6694 |
| Dense matrices | 24 | 18 | 50.1404 | 444.0 | 0.9541 | 1.1060 |
| Reuse old recurrent matrices | 19 | 16 | 2.1395 | 25.6 | 0.5497 | 0.7036 |
| One-bit row quantization | 20 | 18 | 4.0168 | 65.0 | 0.5595 | 0.7133 |
| Two-bit row quantization | 17 | 17 | 5.5168 | 81.4 | 0.5646 | 0.7194 |
| Shared random directions | 20 | 16 | 2.8903 | 40.9 | 0.5526 | 0.7056 |
| Shared matrix-SVD directions | 20 | 18 | 2.8903 | 40.1 | 0.5534 | 0.7084 |
| Shared transported-read directions | 19 | 18 | 2.8903 | 40.7 | 0.5515 | 0.7060 |

The shared read basis reproduced 19/24 dense sequences. Reusing the old matrices reproduced 19/24. The fastest comparator that reproduced all dense sequences was text splice with a prefix checkpoint. Relative to it the candidate's warm complete service time ratio was 0.824 (paired 95% document bootstrap interval 0.780 to 0.867). The composite pilot criterion failed.

Warm service assumes the updated donor cache already exists. Cold service charges its construction. Both times include encoding, wire transfer, reconstruction and answering both questions. Warm update-ready time ends at a usable cache. The shared read basis required 786,495 bytes to install once, and fitting the read and comparison bases took 8.11 seconds in total. The amortization calculation in the score file charges the whole calibration run to the candidate. [Complete service profiles and matched comparisons](summary.json) · [Observed timing and answers](rows.jsonl)

## Reproduction

Standard Python is sufficient to rescore. With the pinned Qwen MLX model, fit both data-dependent bases in a separate working copy before running fresh inference with empty append-only rows. The reference dataset comes from the pinned public read-projection release. [Environment](../read_projection/README.md) · [Source hashes](bundle.json)

```sh
p=experiments/shared_read_basis
python "$p/score.py" .
python "$p/fit.py" "$QWEN9B" .
python "$p/serve.py" "$QWEN9B" .
```
