# What does the complete update cost?

I measured 16 requests from eight previously exposed bAbI and MQuAKE development cases on an Apple M5 with four-bit Qwen3.5-9B. The receiver has the old input and cache and needs to continue generation. Each request uses the same updated text. A sender can transmit a UTF-8 splice, or calculate an updated cache and transmit the changed portions. I charged the attention, convolution and recurrent states, the binary frames, and processing at both ends. Bytes actually crossed a local socket. This measures local socket transport inside one process, using a socket writer thread. The donor and receiver share one physical GPU serially. [Frozen comparison](contract.json) · [Inputs](cases.json)

| Policy | Exact dense sequences / 16 | Mean bytes on wire | Mean seconds, update supplied as text | Mean seconds, donor cache already available |
|---|---:|---:|---:|---:|
| UTF-8 splice and fresh prefill | 16 | 101 | 0.5887 | 0.5887 |
| UTF-8 splice and checkpoint reuse | 16 | 101 | 0.5506 | 0.5506 |
| Dense recurrent state | 16 | 54,299,138 | 0.6295 | 0.3547 |
| Four transported read directions | 16 | 7,113,906 | 0.8433 | 0.5873 |
| Two-bit recurrent delta | 16 | 7,507,906 | 0.5953 | 0.3483 |
| Server answer only | 16 | 17 | 0.5440 | 0.2938 |

The strongest required cold baseline was utf-8 splice and checkpoint reuse. Its mean was 0.5506 seconds versus 0.8433 for projection. Projection required 6.78 MiB per request including framing and the other cache components. A separate server-side answer-only control illustrates the smaller service sufficient when the receiver needs only the known answer text. This control cannot continue locally from the updated state.

The updated prompts averaged 93.0 tokens. I stored old checkpoints every 32 tokens and at the end. The median sum of logical snapshot tensor sizes was 153.2 MiB, before physical buffer sharing. The cold column includes constructing a donor cache when needed. The warm column excludes that step. Both include synchronized answer generation. Models were already loaded, and old-cache setup is reported separately. Prefix replays may use a different batching schedule than fresh prefill. The dense transfer checks exact restoration of all active tensor values. [Per-role accuracy, components and modeled link rates](summary.json) · [All measured requests](rows.jsonl)

The development criterion for a 20% improvement with preserved answers failed.

## Reproduction

Run from the repository root. The model-free scorer uses the Python standard library. Fresh inference needs the pinned MLX model and a separate output directory containing the experiment files with an empty append-only row file. [Model and data revisions](../read_projection/release.json)

```sh
p=experiments/update_service
python "$p/score_service.py" .
python "$p/cache_service.py" "$QWEN9B" .
```
