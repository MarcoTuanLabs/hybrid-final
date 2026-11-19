# Does the finite normalized objective explain answer control at the same search budget?

On the same layer-28 convolution write and four observed query slots, I vary the objective used to design the read. Finite normalization compares the two separately normalized reads. The raw objective measures their unnormalized difference. The first-order objective applies the RMS derivative at the unwritten read to the write-induced displacement, including the derivative’s dependence on the query during optimization. All three use the same finite native-query target, one native start, the 60-degree cone, 16 iterations, and eight proposals per iteration. Two more controls derive their target from their own native contrast. An additional Muir-inspired comparator chooses the exponent of a fractional inverse row-energy gain by the finite objective. The six readers have different arithmetic costs. Their solver times are archived separately. [Muir counterfactual readout interventions](https://arxiv.org/html/2611.08376v1)

I fix the decision thresholds and 32 new seeds before independent inference. The development gate determines whether to consume them. Both formats from a seed stay together in paired testing. All arms use the same physical donor slot. The deployed C++ solver provides an extra check on full-vocabulary predictions. [Protocol](contract.json) · [Independent prompts](prompt_hashes.json) · [Cohort audit](cohort_audit.json)

## Development outcome

| Objective | Correct written target /16 | Controls /48 | Four-slot solver time per request (ms) |
|---|---:|---:|---:|
| Deployed C++ finite solver | 13 | 48 | Unmeasured |
| Finite normalized contrast | 13 | 48 | 27.8 |
| Raw, common target | 0 | 48 | 23.1 |
| First order, common target | 14 | 48 | 28.4 |
| Raw, own target | 0 | 48 | 22.5 |
| First order, own target | 14 | 48 | 25.5 |
| Fractional inverse row-RMS gain | 0 | 48 | 2.4 |

The two first-order alternatives each controlled 14/16 answers and the finite objective controlled 13/16. The raw alternatives and row-gain control each controlled zero. The finite solver reproduced the deployed top token on 64/64 requests (maximum absolute vocabulary-logit difference 0). All methods retained 48/48 controls. The finite objective missed its prespecified promotion rule, so the 32 confirmation seeds remain unused. Solver times cover Python query construction only; the search budget matched iterations and proposals. Reusing activations across arms precludes an end-to-end latency claim.

[Development rows](pilot/rows.jsonl) · [Paired counts](pilot/paired_checks.json) · [Development score](pilot/summary.json) · [Smoke check](smoke/summary.json) · [Independent-cohort registration](https://github.com/MarcoTuanLabs/hybrid-final/tree/b018fe4215ea6d2baa5ce257262deab1b34fde68/experiments/matched_read_objectives)

## Reproduction

Use the pinned four-bit Qwen checkpoint, MLX on Apple Silicon, NumPy and SciPy. Use separate new output directories for smoke, pilot and confirmation.

```sh
p=experiments/matched_read_objectives
bash experiments/quantized_write/build.sh
python "$p/check_math.py"
python "$p/run_matched.py" "$QWEN9B" . "$SCRATCH" pilot
python "$p/score.py" "$SCRATCH" pilot
```
