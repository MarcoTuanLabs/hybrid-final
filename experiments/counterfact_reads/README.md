# Choosing two recurrent heads for a factual revision

I adapted CounterFact to a context update. The recipient sees a fact with its original object. The donor sees the same fact with the prescribed counterfactual object. Each record contributes its canonical sentence stem, its first paraphrase and its first neighborhood prompt. I ask the model to complete each stem and score its first freely generated token, allowing case and surrounding whitespace to vary. Both candidate objects must consist of one alphabetic token. This adaptation measures an immediate response to a supplied revision. ROME originally evaluated weight edits with likelihood-based scores. [Original CounterFact study](https://rome.baulab.info/) · [Authors’ corpus](https://rome.baulab.info/data/dsets/counterfact.json)

I trace the four known chat tokens in the recipient and compute which reads would change if I replaced its recurrent matrices at layer 28. The primary method chooses two adjacent value heads that share a query. It ranks each pair by the squared change in its reads after headwise root-mean-square normalization. In each selected head I fit a rank-one correction to the desired read changes at those four positions. The fit uses the leading singular component of the value-by-position response and the pseudoinverse of the four transported queries. The receiver adds the product of two FP32 vectors to each selected matrix. A pair thus carries 2,048 value bytes plus a pair index. The background trace and full donor are available in this causal experiment.

Raw-response and fixed-pair controls use the same rank-one fit. A third control ranks pairs by matrix norm and applies ordinary rank-one matrix SVD. Larger controls transfer both full matrices of the selected pair, the entire layer’s matrices, all recurrent matrices, or the complete donor document. The gates for pilot success and selection of the 24 subjects were fixed before inference. [Protocol](contract.json) · [Case IDs and hashes](cases.json) · [Source hashes](frozen_files.json)

## Reproduction

Use the pinned four-bit Qwen checkpoint and MLX on Apple Silicon. Download the original CounterFact JSON from the authors and supply an empty output directory.

```sh
p=experiments/counterfact_reads
bash experiments/quantized_write/build.sh
python "$p/check_geometry.py"
python "$p/probe.py" "$QWEN9B" . "$COUNTERFACT" "$SCRATCH"
python "$p/score.py" "$SCRATCH"
```
