# Reproducing the cache-transfer comparison

The scorer runs on a CPU with NumPy and reads all archived per-question rows. From the repository root run the following commands.

```sh
python experiments/read_projection/score.py dev
python experiments/read_projection/score.py test
python experiments/quantized_write/score.py
```

Fresh inference uses Apple Silicon, MLX and a local checkout of the model revision below. The recorded environment used Python 3.13, mlx 0.32.4, mlx-lm 0.32.2, numpy 2.5.4 and scipy 1.18.1. The original machine used its locally installed MLX runtime. The release lists model file hashes, and verifies them before inference. Use separate output directories for each execution. The runner refuses to replace existing rows. [Model revision](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit/tree/8b2b98c00a6b4d291155e4890773ca8f769aee53)

```sh
python experiments/read_projection/run.py test /path/to/Qwen3.5-9B-4bit runs/projection-test
python experiments/read_projection/score.py test --run-dir runs/projection-test
bash experiments/quantized_write/build.sh
python experiments/quantized_write/run.py /path/to/Qwen3.5-9B-4bit runs/write-pilot
python experiments/quantized_write/score.py --run-dir runs/write-pilot
```

The evaluation forms an old and an updated cache separately for each question. Four fixed suffix tokens from the donor supply queries, keys and gates for the projection. The encoder receives no answer tokens. The receiver reconstructs a dense recurrent matrix and generates its own answer. The sender and receiver share donor attention and convolution states. Payload sizes count the recurrent correction alone. [Method and derivation](method.md) · [Data provenance](data/ATTRIBUTION.md)

The held-out phase uses 32 official bAbI test stories and 32 previously unused MQuAKE cases. Each contributes two questions. For MQuAKE this is a held-out selection from the public counterfactual corpus. Frozen contracts and their original path hashes are provided for provenance. The standalone runners resolve the locally supplied artifacts from this release. [Evidence](../../evidence/read_projection)
