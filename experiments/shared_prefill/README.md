# Does sharing prefill work preserve the memory write?

Two branches of a 9B synthetic retrieval request share part of their prompt. I tested four schedules for preparing the original and donor cache. Each used the same layer-28 intervention, read the same four suffix slots with the 16-step solver, and generated a top token from the full vocabulary. Measurements include encoding, both prefills, applying the write and answering. The eight exposed development seeds produced 64 requests per schedule. [Frozen design](contract.json)

The parity columns count requests on which all 256 coordinates matched the separate full-branch reference. The original-slot check uses the recipient before the edit. All 48 controls consist of 16 written nuisance, 16 unwritten target and 16 unwritten nuisance presentations.

| Schedule | Exact donor /64 | Exact original slot /64 | Same top token /64 | Controlled targets | Correct controls | Complete mean, s |
|---|---:|---:|---:|---:|---:|---:|
| Separate full branches | 64 | 64 | 64 | 13/16 | 48/48 | 0.4769 |
| Separate, truncated donor | 64 | 64 | 64 | 13/16 | 48/48 | 0.4550 |
| Shared prefix, truncated donor | 0 | 0 | 63 | 14/16 | 48/48 | 0.5851 |
| Joint two-row prefill | 0 | 0 | 63 | 14/16 | 48/48 | 0.4151 |

The separately truncated donor reproduced the original write, designed query and output on every request. It reduced mean complete time by 4.6% (paired 95% seed bootstrap interval 2.7% to 6.2%). Joint prefilling reduced mean time by 13.0% (11.3% to 14.4%). It changed donor coordinates on every request. Sharing the longest prefix also changed donor coordinates on every request and increased mean time by 22.7%. Both schedules controlled one additional target on these exposed seeds. Neither met the prespecified same-write and cost conditions. [All results](summary.json) · [Raw per-request comparisons](rows.jsonl)

The nominal number of full layer-token evaluations over 32 distinct prompt pairs was 208,256 for separate or joint full branches, 195,240 for truncation, and 164,272 for sharing with truncation. Truncation additionally evaluated 3,254 partial layer-28 QKV token positions, or 2,148 after prefix sharing. The common four-token suffix adds 4,096 layer-token positions per method. These counts describe model work before kernel scheduling and lazy evaluation. Device time is measured separately.

## Reproduction

Score the archive with standard Python. Fresh inference uses the pinned weights, MLX and the compiled reader. Move the archived rows out of a disposable checkout before timing. [Reader setup](../jacobian_service/README.md) · [Source and evidence hashes](bundle.json)

```sh
p=experiments/shared_prefill
python "$p/score.py"
bash experiments/quantized_write/build.sh
python "$p/compare_prefix.py" "$QWEN9B" .
```
