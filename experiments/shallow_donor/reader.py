"""Approximate a deep donor slot from an early paired hidden-state difference."""
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
MODES = ("truncated", "joint", "absolute", "paired")
rng = np.random.default_rng(518907)
SIGNS = rng.choice(np.array([-1, 1], np.float32), size=4096)
KEEP = np.sort(rng.choice(4096, size=384, replace=False))


def project(vectors):
    hidden = np.asarray(vectors.astype(mx.float32)) * SIGNS
    for width in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        pairs = hidden.reshape(4, -1, 2, width)
        a, b = pairs[:, :, 0], pairs[:, :, 1]
        hidden = np.stack((a+b, a-b), axis=2).reshape(4, 4096)
    return hidden[:, KEEP] / 64


def edited_positions(a, b):
    assert len(a) == len(b)
    changed = [i for i, (left, right) in enumerate(zip(a, b)) if left != right]
    assert len(changed) == 2
    return changed


def run_branch(trial, ids, changed, stop):
    net = trial.model.model
    cache = make_prompt_cache(trial.model)
    h = net.embed_tokens(mx.array([ids]))
    masks = (create_attention_mask(h, cache[net.fa_idx]), create_ssm_mask(h, cache[net.ssm_idx]))
    for i, layer in enumerate(net.layers[:stop]):
        h = layer(h, mask=masks[int(layer.is_linear)], cache=cache[i])
        if i == 11:
            features = mx.stack((h[0,-1], h[0,-8:].mean(axis=0), h[0,changed[0]], h[0,changed[1]]))
    if stop == 32:
        slot = cache[28][0][:, -1:, trial.lo:trial.lo+256]
        mx.eval(*[layer.state for layer in cache], features)
    elif stop == 28:
        layer = net.layers[28]
        slot = layer.linear_attn.in_proj_qkv(layer.input_layernorm(h))[:, -1:, trial.lo:trial.lo+256]
        mx.eval(slot, features)
    else:
        assert stop == 12
        slot = None
    return cache, slot, project(features)


def fit(x, y, centered):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(0) if centered else np.zeros_like(x[0])
    offset = y.mean(0) if centered else np.zeros_like(y[0])
    x -= mean
    scale = np.sqrt(np.mean(x*x, axis=(0,2))).clip(1e-8)
    design = (x / scale[:,None]).reshape(len(x), -1)
    gram = design @ design.T
    gram.flat[::len(gram)+1] += 0.02 * np.trace(gram)/len(gram)
    weights = design.T @ np.linalg.solve(gram, y-offset)
    return dict(mean=mean.astype("f"), scale=scale.astype("f"), weights=weights.astype("f"), offset=offset.astype("f"))


def predict(x, params):
    transformed = (x-params["mean"]) / params["scale"][:,None]
    return transformed.ravel() @ params["weights"] + params["offset"]


def setup(args):
    sys.path.insert(0, str(args.public/"experiments/quantized_write"))
    from run import Trial, make_example
    sys.path.insert(0, str(args.public/"experiments/shared_prefill"))
    from compare_prefix import branch, set_write
    sys.path.insert(0, str(args.public/"experiments/jacobian_service"))
    from runtime import stream
    pins=json.loads((args.public/"experiments/read_projection/release.json").read_text())
    for name, digest in pins["model_files"].items():
        with (args.model/name).open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest() == digest
    cfg = json.loads((args.public/"evidence/quantized_9b/contract.json").read_text())
    trial=Trial(args.model,cfg)
    mx.set_cache_limit(96*1024**2)
    return trial, make_example, branch, set_write, stream


def calibrate(trial, make):
    start=time.monotonic()
    xa, xb, ya, yb = [], [], [], []
    for seed in range(38455000,38455048):
        for surface in ("table","prose"):
            for nuisance in (False,True):
                ex=make(seed,surface,nuisance)
                a,_=trial.encode(ex,"a")
                b,_=trial.encode(ex,"b")
                positions=edited_positions(a,b)
                for ids,features,outputs in ((a,xa,ya),(b,xb,yb)):
                    _,slot,x=run_branch(trial,ids,positions,28)
                    features.append(x)
                    outputs.append(np.asarray(slot.astype(mx.float32)).reshape(256))
        if seed%8==7:
            print(seed,round(time.monotonic()-start,1),flush=True)
        if time.monotonic()-start>480:raise TimeoutError("calibration")
    both=fit(np.concatenate((xa,xb)), np.concatenate((ya,yb)),True)
    paired=fit(np.array(xb)-xa,np.array(yb)-ya,False)
    params={m+"_"+k:v for m,p in (("absolute",both),("paired",paired)) for k,v in p.items()}
    np.savez(HERE/"params.npz",**params)
    report=dict(pairs=len(xa),elapsed_s=time.monotonic()-start,coef_bytes=sum(x.nbytes for x in params.values()),
                checkpoint_sha256=hashlib.sha256((HERE/"params.npz").read_bytes()).hexdigest())
    (HERE/"calibration.json").write_text(json.dumps(report,indent=2)+"\n")


def evaluate(phase, trial, make, branch, set_write, stream):
    if phase == "test":
        assert json.loads((HERE/"dev_summary.json").read_text())["promote"]
    cal=json.loads((HERE/"calibration.json").read_text())
    assert hashlib.sha256((HERE/"params.npz").read_bytes()).hexdigest()==cal["checkpoint_sha256"]
    with np.load(HERE/"params.npz") as z:
        weights={m:{k:z[m+"_"+k] for k in ("weights","mean","scale","offset")} for m in ("absolute","paired")}
    def service(ex,mode,active):
        mx.synchronize()
        t=time.perf_counter()
        a,tail=trial.encode(ex,"a")
        b,tail2=trial.encode(ex,"b")
        assert tail==tail2
        if mode in weights:
            positions=edited_positions(a,b)
            base,original,x=run_branch(trial,a,positions,32)
            _,_,y=run_branch(trial,b,positions,12)
            estimate=predict(y-x if mode=="paired" else y, weights[mode])
            w=mx.array(estimate).reshape(1,1,256)
            if mode=="paired":w=w+original.astype(mx.float32)
            w=w.astype(original.dtype)
        else:
            base,w=branch(trial,a,b,"cut" if mode=="truncated" else "joint")
            original=base[28][0][:,-1:,trial.lo:trial.lo+256]
        edit=set_write(trial,base,w)
        prefix=time.perf_counter()-t
        logits,_=stream(trial,base,edit,tail,active,"batch_observed")
        top=int(np.argmax(logits))
        mx.synchronize()
        seconds=time.perf_counter()-t
        flatten=lambda a:np.asarray(a.astype(mx.float32)).ravel()
        return dict(top=top,seconds=seconds,prefix_s=prefix),flatten(w),flatten(original)
    seeds=range(27112000,27112008) if phase=="dev" else range(38456000,38456032)
    for m in MODES:service(make(27112000,"table",False),m,True)
    output=HERE/(phase+"_rows.jsonl")
    seen={tuple(json.loads(l)["key"]) for l in output.read_text().splitlines()} if output.exists() else set()
    start=time.monotonic()
    with output.open("a") as f:
        for seed in seeds:
            for surface in ("table","prose"):
                for nuisance in (False,True):
                    ex=make(seed,surface,nuisance)
                    recipient,donor=[trial.tok.encode(ex[k],add_special_tokens=False)[0] for k in ("value_a","value_b")]
                    for active in (False,True):
                        key=(seed,surface,nuisance,active)
                        if key in seen:continue
                        order=list(MODES)
                        random.Random(941*seed+17*nuisance+31*active+7*(surface=="prose")).shuffle(order)
                        res={m:service(ex,m,active) for m in order}
                        _,exact,original=res["truncated"]
                        delta=exact-original
                        arms={}
                        for m,(row,w,old) in res.items():
                            d=w-old
                            arms[m]=row|dict(slot_rmse=float(np.sqrt(np.mean((exact-w)**2))),
                                relative_delta_error=float(np.linalg.norm(d-delta)/max(np.linalg.norm(delta),1e-10)),
                                delta_cosine=float(d@delta/max(np.linalg.norm(d)*np.linalg.norm(delta),1e-10)))
                        f.write(json.dumps(dict(key=key,recipient=recipient,donor=donor,arms=arms))+"\n")
                        f.flush()
            print(seed,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>(480 if phase=="dev" else 1440):raise TimeoutError(phase)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("calibrate","dev","test"))
    parser.add_argument("model",type=Path)
    parser.add_argument("public",type=Path)
    a=parser.parse_args()
    trial,make,branch,set_write,stream=setup(a)
    if a.phase=="calibrate":calibrate(trial,make)
    else:evaluate(a.phase,trial,make,branch,set_write,stream)
