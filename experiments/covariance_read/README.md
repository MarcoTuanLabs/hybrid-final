# Testing curvature-conditioned queries on the same memory write

I compared a prefix-key covariance correction with a reader optimized for the finite effect of a specified write after headwise normalization. The frozen four-bit Qwen3.5-9B checkpoint received exactly the original 256-coordinate convolution update. The candidate queries had the same norm and four available read slots. Two controls used fixed contraction strengths. A third selected the contraction strength separately for each slot using the actual normalized-write objective. A fourth used the known source key and whitened it with the regularized prefix-key covariance. [Read geometry and angular bound](method.md) · [Prospective design](contract.json)

All seven methods ran on both the written and unwritten branches and produced full-vocabulary predictions. The two 4-bit prompt formats share a seed. Each seed supplies two written target cases and six preservation cases. The original curvature-conditioned models learn a gate during pretraining. This experiment compares inference-time adaptations to the same frozen Qwen model. [Original CCQ paper](https://arxiv.org/abs/2606.01294)

## Previously evaluated development seeds

| Reader | Controlled targets | Preserved controls | Median query rotation |
|---|---:|---:|---:|
| Native query | 0/32 | 96/96 | 0.00° |
| Half-strength contraction | 0/32 | 96/96 | 0.03° |
| Full-strength contraction | 0/32 | 96/96 | 0.06° |
| Write-informed contraction | 0/32 | 96/96 | 0.00° |
| Source key whitened by prefix keys | 0/32 | 96/96 | 89.73° |
| Source key whitened by state Gram | 15/32 | 96/96 | 89.87° |
| Optimized normalized read | 26/32 | 96/96 | 25.67° |

Relative to half-strength contraction, the optimized reader gained 26 target answers and lost 0. The paired gain was 81.2 percentage points (95% bootstrap interval 62.5 to 100.0, resampling whole seeds). The strength-one contraction rotated the native query at most 0.08 degrees over these slots. The comparison uses shared geometry preparation and tests answer quality. [All paired comparisons](dev_summary.json) · [Per-prompt outputs](dev_rows.jsonl)

Against the state-Gram whitened-key reader, the normalized reader gained 11 targets and lost 0. The paired gain was 34.4 percentage points (95% seed-bootstrap interval 15.6 to 53.1).

## Prospective confirmation on new seeds

| Reader | Controlled targets | Preserved controls | Median query rotation |
|---|---:|---:|---:|
| Native query | 0/64 | 192/192 | 0.00° |
| Half-strength contraction | 0/64 | 192/192 | 0.03° |
| Full-strength contraction | 0/64 | 192/192 | 0.06° |
| Write-informed contraction | 0/64 | 192/192 | 0.00° |
| Source key whitened by prefix keys | 0/64 | 192/192 | 89.75° |
| Source key whitened by state Gram | 30/64 | 192/192 | 89.89° |
| Optimized normalized read | 57/64 | 192/192 | 25.70° |

Relative to half-strength contraction, the optimized reader gained 57 target answers and lost 0. The paired gain was 89.1 percentage points (95% bootstrap interval 78.1 to 96.9, resampling whole seeds). The strength-one contraction rotated the native query at most 0.09 degrees over these slots. The comparison uses shared geometry preparation and tests answer quality. [All paired comparisons](test_summary.json) · [Per-prompt outputs](test_rows.jsonl)

Against the state-Gram whitened-key reader, the normalized reader gained 30 targets and lost 3. The paired gain was 42.2 percentage points (95% seed-bootstrap interval 23.4 to 60.9).

## Reproduction

Run the standard-library scorer for each archived phase. For a fresh Apple Silicon run in a disposable checkout, first compile the existing native-start solver and move the archived phase observations out of the experiment directory. Supply the pinned MLX checkpoint and use the repository root as the last argument. The program opens the confirmation phase only after the fixed development condition passes. [Model pins](../read_projection/release.json) · [Data separation](split.json) · [Artifact hashes](bundle.json)

```sh
p=experiments/covariance_read
python "$p/score.py" dev
python "$p/verify_split.py" .
bash experiments/quantized_write/build.sh
python "$p/compare.py" dev "$QWEN9B" .
python "$p/score.py" dev
```

The confirmation commands are `python "$p/compare.py" test "$QWEN9B" .` and `python "$p/score.py" test`.
