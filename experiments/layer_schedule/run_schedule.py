"""Schedule independent recipient and donor graphs without changing either input shape."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared_prefill"))
from compare_prefix import branch, set_write

MODES = ("full", "cut", "lazy", "one", "four")


def interleave(trial, recipient_ids, donor_ids, mode):
    net = trial.model.model
    recipient, donor = [make_prompt_cache(trial.model) for _ in range(2)]
    x, y = [net.embed_tokens(mx.array([ids])) for ids in (recipient_ids, donor_ids)]
    masks = [(create_attention_mask(h, c[net.fa_idx]), create_ssm_mask(h, c[net.ssm_idx]))
             for h, c in ((x, recipient), (y, donor))]
    block = {"lazy": 0, "one": 1, "four": 4}[mode]
    for i, layer in enumerate(net.layers):
        linear = int(layer.is_linear)
        x = layer(x, mask=masks[0][linear], cache=recipient[i])
        if i < 28:
            y = layer(y, mask=masks[1][linear], cache=donor[i])
            if block and (i+1) % block == 0:
                mx.async_eval(x, y)
        elif i == 28:
            values = layer.linear_attn.in_proj_qkv(layer.input_layernorm(y))
            write = values[:, -1:, trial.lo:trial.lo+256]
    mx.eval(write, *[c.state for c in recipient])
    return recipient, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("public", type=Path)
    a = parser.parse_args()
    repo = a.public.resolve()
    sys.path.insert(0, str(repo / "experiments/quantized_write"))
    from run import Trial, make_example
    sys.path.insert(0, str(HERE.parent / ("jacobian_service" if (HERE.parent / "jacobian_service/runtime.py").exists() else "stream9")))
    from runtime import stream
    manifest = json.loads((repo/"experiments/read_projection/release.json").read_text())
    for name, digest in manifest["model_files"].items():
        with (a.model/name).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == digest
    config = json.loads((repo/"evidence/quantized_9b/contract.json").read_text())
    trial = Trial(a.model, config)

    def run(example, mode, active):
        mx.synchronize()
        start = time.perf_counter()
        recipient, tail = trial.encode(example, "a")
        donor, other_tail = trial.encode(example, "b")
        assert tail == other_tail
        if mode in ("full", "cut"):
            base, w = branch(trial, recipient, donor, mode)
        else:
            base, w = interleave(trial, recipient, donor, mode)
        edited = set_write(trial, base, w)
        split = time.perf_counter()
        logits, q = stream(trial, base, edited, tail, active, "batch_observed")
        top = int(np.argmax(logits))
        mx.synchronize()
        end = time.perf_counter()
        def digest(tensor):
            return hashlib.sha256(np.asarray(tensor.view(mx.uint16)).tobytes()).hexdigest()
        row = dict(seconds=end-start, prefix_s=split-start, top=top, write=digest(w),
                   original=digest(base[28][0][:, -1:, trial.lo:trial.lo+256]),
                   queries=hashlib.sha256(np.asarray(q).tobytes()).hexdigest())
        return row, logits

    for m in MODES:
        run(make_example(27112000, "table", False), m, True)
    start = time.monotonic()
    dst = HERE/"rows.jsonl"
    seen = {tuple(json.loads(line)["key"]) for line in dst.read_text().splitlines()} if dst.exists() else set()
    with dst.open("a") as out:
        for seed in range(27112000, 27112008):
            for surface in ("table", "prose"):
                for nuisance in (False, True):
                    ex = make_example(seed, surface, nuisance)
                    for active in (False, True):
                        key = (seed, surface, nuisance, active)
                        if key in seen:
                            continue
                        modes = list(MODES)
                        random.Random(53*seed + 7*active + 11*nuisance + 17*(surface == "prose")).shuffle(modes)
                        results = {m: run(ex, m, active) for m in modes}
                        ref_logit = results["full"][1]
                        arms = {m: row | dict(max_logit_diff=float(np.max(abs(logits-ref_logit))))
                                for m, (row, logits) in results.items()}
                        labels = {p: trial.tok.encode(ex[k], add_special_tokens=False)[0]
                                  for p,k in (("recipient","value_a"),("donor","value_b"))}
                        out.write(json.dumps(dict(key=key, arms=arms, **labels))+"\n")
                        out.flush()
            print(seed, round(time.monotonic()-start, 1), flush=True)
            mx.clear_cache()
            if time.monotonic()-start > 480:
                raise TimeoutError("Layer schedule budget")


if __name__ == "__main__":
    main()
