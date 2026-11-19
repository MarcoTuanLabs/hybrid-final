# Donor-free changes to local KV cache entries

I tested whether the four attention rows identified by the local donor experiment can be repaired using only the old cached document and the two country positions being exchanged. A row combines country identity with its contextual role. Moving the entire old value to its new position can carry information about the wrong person. I estimated the part of a source–target difference that predicts how the model would rewrite that row.

During calibration I generated both document orders on 96 additional prompt IDs with the authors' two templates. No future questions, answer tokens or labels entered calibration. At each layer, key/value head and token offset, I removed rotary phase from keys and subtracted the mean for the corresponding template and entity position. I obtained a 32-dimensional output basis from the actual counterfactual cache differences. A ridge map predicts the coefficients of the new target's displacement from the two original centered rows. The server adds this displacement to the old target, restores its key phase, and applies the same rule in the reverse direction at the other position. All recurrent matrices and other KV rows remain in place. [Frozen protocol](contract.json) · [Causal localization](../binding_cache/README.md)

An identity round-trip on real caches exposed an invalid negative-offset rotary call. I replaced it with an FP32 reflection around a positive rotation, refitted the model and reran all comparisons. [Engineering correction](repair.json) · [Inverse rotation](../halumem_bridge/cache_rotary.py)

The simpler projection uses the same counterfactual directions and takes ordinary inner products with the centered difference. The centered exchange omits the projection. Another control reruns each edited country and its punctuation from the original checkpoint immediately preceding that country. For the second edit this checkpoint contains the old version of the earlier country. The four-row donor oracle has access to the exact updated document. The evaluation shares prefills and measures answer quality.

| Update | Correct changed | Correct stable | Changed successes lost vs. full | Previously correct stable answers lost |
|---|---:|---:|---:|---:|
| Old document | 0/48 | 19/24 | 40 | 0 |
| Complete updated document | 40/48 | 19/24 | 0 | 1 |
| Correction in text | 28/48 | 15/24 | 12 | 4 |
| Exchange old values | 27/48 | 18/24 | 13 | 1 |
| Exchange centered old keys and values | 33/48 | 19/24 | 10 | 0 |
| Project onto the paired edit subspace | 36/48 | 18/24 | 6 | 1 |
| Predict paired edit coefficients | 35/48 | 18/24 | 7 | 1 |
| Recompute two tokens at each old checkpoint | 30/48 | 18/24 | 10 | 1 |
| Use the donor's true local K/V | 40/48 | 19/24 | 2 | 0 |

The paired estimator controlled 35/48 changed answers and retained 18/19 previously correct stable answers. Compared with the orthogonal projection it gained 0 and lost 1 changed answers. The declared promotion condition failed. The original fit took 70.04 seconds and stored 9,175,040 bytes in means, bases and maps. This fit is specific to the three-entity format. [All outcomes](dev_rows.jsonl) · [Paired scores](dev_summary.json) · [Calibration receipt](fit.json)

Feng and Steinhardt identified binding vectors by comparing contextual activations. CacheBlend selectively recomputes tokens to repair stale attention caches. Here the calibration target is the paired native K/V displacement at a factual transposition. [Binding ID](https://arxiv.org/abs/2310.17191) · [CacheBlend](https://arxiv.org/abs/2405.16444)

## Reproduction

Use the standard-library scorer to read the archived outputs. Fit the coefficients anew from the pinned model and the authors' datasets for fresh inference. Move the archived development rows out of a disposable checkout first.

```sh
p=experiments/binding_transport
python "$p/score.py"
python experiments/binding_transfer/inputs.py "$BINDING_DATA"
python "$p/run_transfer.py" calibrate "$BINDING_DATA" "$QWEN9B" .
python "$p/run_transfer.py" dev "$BINDING_DATA" "$QWEN9B" .
python "$p/score.py"
```

[Model and environment](../read_projection/README.md) · [Artifact hashes](bundle.json)
