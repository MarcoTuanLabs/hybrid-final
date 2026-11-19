# Locating changed country bindings on independent documents

I fixed the methods, analysis and 32 unused CAPITALS prompt IDs in a public commit before confirmation. The 64 documents (two templates each) did not overlap the 440 earlier or reserved documents. Each document exchanges two countries. After the edit the model answers one question about each of the three people independently from the same cache. All 128 changed questions and all 64 stable questions count. The two templates share a seed in the paired bootstrap. [Registration before inference](https://github.com/MarcoTuanLabs/hybrid-final/tree/d3dcb1407ac2fce0a62f08642f0c01d7ebbcd10c/experiments/binding_confirmation) · [Pinned task and source](../binding_transfer/README.md)

The local intervention transfers the donor keys and values at the two country tokens and their following punctuation in all eight full-attention layers. The largest-change controls use the same four-position budget and measure squared K/V differences over layers and heads before seeing any question. One can select any four positions, while the other excludes the four lexical positions. The recurrent intervention transfers every recurrent matrix and convolution history. Query redirection runs the model on the old cache with the name of the original holder of the requested country in both places in the question prefix. The stable query uses its original name. The complete vocabulary logits of every redirect agreed with an independent calculation on the corresponding original question.

| Method | Correct changed capitals | Correct stable capitals | Changed, both native prompts correct | Local gains vs. method | Local losses vs. method |
|---|---:|---:|---:|---:|---:|
| Old document | 0/128 | 59/64 | 0/81 | 98 | 0 |
| Full updated document | 105/128 | 60/64 | 81/81 | 3 | 10 |
| All updated recurrent and convolution states | 0/128 | 60/64 | 0/81 | 98 | 0 |
| Updated keys and values at the four lexical positions | 98/128 | 60/64 | 71/81 | 0 | 0 |
| Updated values at the four lexical positions | 92/128 | 60/64 | 67/81 | 7 | 1 |
| Four largest changes in the attention cache | 96/128 | 60/64 | 70/81 | 2 | 0 |
| Four largest changes outside the lexical positions | 0/128 | 58/64 | 0/81 | 98 | 0 |
| Append the two corrected statements | 75/128 | 55/64 | 56/81 | 29 | 6 |
| Redirect the question on the original cache | 99/128 | 59/64 | 72/81 | 2 | 3 |

The local transplant retained 95/105 of the full replay's changed successes. It gained 3 answers and lost 10 relative to full replay. It lost 0 stable answers previously correct in the old document. Compared with transplanting the recurrent states alone, the gain on changed capitals was 76.6 percentage points (95% paired seed-bootstrap interval 67.2 to 85.2). The preregistered localization criterion passed. The unrestricted energy selector chose 193/256 positions from the lexical set and exactly matched it on 10/64 documents. [All paired comparisons](summary.json) · [Individual outputs and selected sites](rows.jsonl)

These donor transplants need the updated document cache. They establish where bindings can be recovered for this model, task and update. Query redirection is a structured alternative with access to the known permutation. The benchmark generator supplies the person/country association and a static country/capital table. Full deployment cost and performance on free-form factual edits require different measurements.

## Reproduction

The archived scorer uses Python's standard library. Regenerating the split and inference requires the source benchmark and the pinned model. For fresh inference work in a disposable checkout and move the archived rows aside before running.

```sh
p=experiments/binding_confirmation
python "$p/score.py"
python experiments/binding_transfer/inputs.py "$BINDING_DATA"
python "$p/confirm.py" split "$BINDING_DATA" "$QWEN9B" .
python "$p/confirm.py" run "$BINDING_DATA" "$QWEN9B" .
python "$p/score.py"
```

[Artifact hashes](bundle.json) · [Four-bit model revision](../read_projection/release.json)
