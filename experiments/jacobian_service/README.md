# Reading the same native memory write at lower cost

I changed 256 coordinates of the layer-28 convolution history in four-bit Qwen3.5-9B. I fixed the two affected heads, four read slots, query norms, 60-degree angular budget and output vocabulary. The donor computation can stop at the layer-28 QKV projection that supplies the edit. I compared the existing 16-step solver with four steps, an analytic maximizer of the first-order normalized objective and an unmodified query. The one-step method kept the better of the proposed and original queries under the finite objective. [Frozen design](contract.json)

I timed both prompt prefills, the write, the readout design and the output token. Each method served the same 64 requests independently after warm-up, in shuffled order on an Apple M5. The eight previously exposed seeds each supply two prompt formats and four conditions. The two unwritten conditions have 16 cases each. [Outputs, query hashes, constraints and timings](rows.jsonl)

| Method | Target | Nuisance | Unwritten | Prefills and write, s | Complete mean, s |
|---|---:|---:|---:|---:|---:|
| Full donor, 16 steps | 13/16 | 16/16 | 32/32 | 0.4659 | 0.5308 |
| Truncated donor, 16 steps | 13/16 | 16/16 | 32/32 | 0.4528 | 0.5186 |
| Truncated donor, four steps | 3/16 | 16/16 | 32/32 | 0.4464 | 0.5106 |
| Truncated donor, one Jacobian step | 0/16 | 16/16 | 32/32 | 0.4455 | 0.5104 |
| Truncated donor, native query | 0/16 | 16/16 | 32/32 | 0.4398 | 0.4998 |

Stopping the donor early preserved every write byte, every designed query and all 64 top tokens. It reduced complete time by 2.3% relative to the full donor (paired 95% seed bootstrap interval for the reduction, 0.4% to 4.2%). Under the truncated donor, both approximations lost target answers. Removing query optimization altogether saved 3.6% on this implementation and controlled zero target answers. These timings give little incentive to train a query predictor under the same service conditions. [Paired results](summary.json)

Across the 32 distinct prompt pairs, 1,106 positions were shared out of 6,636 positions in the two prefills and four-token suffix. Wall time and the arithmetic required per position depend on the schedule. [Token counts](prefix_budget.json)

## Reproduction

Python's standard library can rescore the archived requests. Fresh inference uses the pinned 4-bit model, MLX and the compiled solver. Run the following commands from the repository root on Apple Silicon. Move the archived rows out of a disposable checkout before fresh timing. The evaluation otherwise resumes from the rows it finds. [Model and input hashes](../read_projection/release.json) · [Code and evidence hashes](bundle.json)

```sh
p=experiments/jacobian_service
python "$p/score.py"
bash experiments/quantized_write/build.sh
python "$p/prefix_budget.py" "$QWEN9B" .
python "$p/compare.py" "$QWEN9B" .
```
