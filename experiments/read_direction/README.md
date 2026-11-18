# Which part of the recurrent read changes the answer?

I tested the local value write in the four-bit Qwen3.5-9B model. The donor and recipient differ in two value heads at layer 28. For each of the four allowed suffix tokens, I ran the same optimized query on the recipient and on the written state. The resulting two head vectors are the inputs to gated RMS normalization. I independently exchanged their directions and their Euclidean lengths, supplied the new vectors before normalization, and let the model finish normally. Both vectors were nonzero in every tested head. The rest of the model and the stored write were held fixed. [Original quantized value-write experiment](../../evidence/quantized_9b) · [Registered hypothesis and decision rules](contract.json)

With direction alone, each written vector is rescaled to the length of the unwritten vector. With length alone, each unwritten vector is rescaled to the length of its written counterpart. I also restored the complete unwritten vector in both heads as a control. I compared every vocabulary logit from that reset with a genuine unwritten rollout. This design tests a causal role for the read components of this particular sparse write. Angular and radial activation interventions have precedents. [Related geometric interventions](https://arxiv.org/abs/2606.06735) · [Implementation](factorial.py)

| Condition on the two written heads | Development target / 32 | Fresh target / 128 | Fresh nuisance preserved / 128 |
|---|---:|---:|---:|
| Native query with the write | 0 | 0 | 128 |
| Whitened-key query with the write | 15 | 68 | 128 |
| Optimized query without a write | 0 | 0 | 128 |
| Optimized query with the full write | 26 | 99 | 128 |
| Written direction, unwritten length | 25 | 95 | 128 |
| Unwritten direction, written length | 0 | 0 | 128 |
| Full unwritten read on the written trajectory | 0 | 0 | 128 |

The 16 previously measured development seeds each supplied a table and a prose presentation. Direction alone retained 25/26 full-write successes. Its target success rate exceeded the length-only arm by 78.1 percentage points (paired 95% seed bootstrap interval 56.2 to 93.8). The largest absolute vocabulary-logit discrepancy after restoring both unedited head reads was 0. The optimized unwritten query preserved 64/64 recipient answers. [All dev outcomes and paired differences](dev_summary.json)

The 64 additional fixed seeds each supplied a table and a prose presentation. Direction alone retained 95/99 full-write successes. Its target success rate exceeded the length-only arm by 74.2 percentage points (paired 95% seed bootstrap interval 64.1 to 83.6). The largest absolute vocabulary-logit discrepancy after restoring both unedited head reads was 0. The optimized unwritten query preserved 256/256 recipient answers. [All test outcomes and paired differences](test_summary.json)

Direction alone lost four of the 99 full-write successes and gained none. Its paired success rate was 3.1 percentage points lower than the full write (95% seed bootstrap interval 0.8 to 6.2 points lower). The independent confirmation met all prespecified criteria. See the [frozen decision rule](contract.json).

The intervention was restricted to the two value heads and four suffix positions used for the original local 9B value write. The 27B write addresses a different mechanism. The prefill and query-design work was shared across controls in this diagnostic. [Data provenance and fresh seed check](seed_check.json)

## Reproduction

Run from the repository root. Scoring uses the Python standard library. Fresh inference uses the pinned MLX model and the compiled solvers from the quantized-write experiment. The test-phase runner checks the development gate and the frozen analysis files. Use a fresh copy without the archived append-only rows when rerunning inference. [Model and build requirements](../../evidence/quantized_9b/contract.json)

```sh
p=experiments/read_direction
python "$p/score_direction.py" dev
python "$p/score_direction.py" test
bash experiments/quantized_write/build.sh
python "$p/factorial.py" dev "$QWEN9B" --public-repo .
python "$p/score_direction.py" dev
python "$p/factorial.py" test "$QWEN9B" --public-repo .
python "$p/score_direction.py" test
```
