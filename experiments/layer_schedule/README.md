# Layer scheduling with a fixed memory write

I interleaved the full recipient and truncated donor through their first 28 common layers. Each branch kept its separate single-row tensor shape. I tested a lazy joint graph and asynchronous dispatch every layer or every four layers. Each schedule charged the two prompt encodings, model states, 256-coordinate write, 16-step readout and output. Methods were timed independently in randomized order on 64 requests from eight exposed development seeds. All used the same pinned 4-bit Qwen3.5-9B weights on an Apple M5. [Frozen design](contract.json)

The four agreement columns count requests that match the separate full-branch reference. The query comparison covers all four designed vectors per request. Each schedule also has three sets of 16 preservation controls.

| Schedule | Exact donor /64 | Exact original slot /64 | Same queries /64 | Same top /64 | Controlled targets | Complete mean, s |
|---|---:|---:|---:|---:|---:|---:|
| Separate full branches | 64 | 64 | 64 | 64 | 13/16 | 0.6022 |
| Separate, truncated donor | 64 | 64 | 64 | 64 | 13/16 | 0.5785 |
| Interleaved, one graph | 64 | 64 | 64 | 64 | 13/16 | 0.5689 |
| Interleaved, dispatch each layer | 64 | 64 | 64 | 64 | 13/16 | 0.5849 |
| Interleaved, dispatch each four layers | 64 | 64 | 64 | 64 | 13/16 | 0.5784 |

The fastest interleaved schedule was interleaved, one graph. Its complete time ratio relative to the original full reader was 0.945 (paired 95% seed bootstrap interval 0.930 to 0.961). The prespecified 20% cost and exact-parity criterion failed. [All answers, controls and intervals](summary.json) · [Timed observations and byte hashes](rows.jsonl)

## Reproduction

The standard-library scoring script reproduces the archived comparisons. After moving archived rows out of a disposable checkout, rerun the timed service with the pinned model on Apple Silicon. [Reader setup](../jacobian_service/README.md) · [Evidence hashes](bundle.json)

```sh
p=experiments/layer_schedule
python "$p/score.py"
bash experiments/quantized_write/build.sh
python "$p/run_schedule.py" "$QWEN9B" .
```
