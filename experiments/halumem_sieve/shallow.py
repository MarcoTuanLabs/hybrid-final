"""Measure the isolated contribution of a turn's values at a shallow recurrent layer."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache
import mlx_lm.models.qwen3_5 as qwen
import numpy as np

HERE=Path(__file__).resolve().parent


def first_layers(model,cache,tokens,probe=None):
    net=model.model
    h=net.embed_tokens(mx.array([tokens]))
    mask=create_ssm_mask(h,cache[0])
    for i in range(2):h=net.layers[i](h,mask=mask,cache=cache[i])
    orig=qwen.gated_delta_update
    record={}
    def read(q,k,v,a,b,al,bias,state,mask,**kw):
        output,end=orig(q,k,v,a,b,al,bias,state,mask,**kw)
        if probe is None:
            record["probe"]=mx.repeat(q[0,-4:],v.shape[2]//q.shape[2],axis=1).astype(mx.float32)
        else:
            _,muted=orig(q,k,mx.zeros_like(v),a,b,al,bias,state,mask,**kw)
            def state_read(s):
                return mx.einsum("hvk,thk->thv",s[0].astype(mx.float32),probe)
            w,z=state_read(end),state_read(muted)
            def norm(x):
                return x*mx.rsqrt(mx.mean(x*x,axis=-1,keepdims=True)+1e-6)
            raw=mx.sqrt(mx.mean((w-z)**2,axis=-1))
            delta=mx.sqrt(mx.mean((norm(w)-norm(z))**2,axis=-1))
            record["scores"]=mx.stack([mx.mean(delta),mx.mean(raw)])
        return output,end
    qwen.gated_delta_update=read
    try:net.layers[2](h,mask=mask,cache=cache[2])
    finally:qwen.gated_delta_update=orig
    mx.eval(*record.values(),*[c.state for c in cache[:3]])
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("model",type=Path);ap.add_argument("public",type=Path);a=ap.parse_args()
    pins=json.loads((a.public/"experiments/read_projection/release.json").read_text())
    for n,h in pins["model_files"].items():
        with (a.model/n).open("rb") as fp:assert hashlib.file_digest(fp,"sha256").hexdigest()==h
    model,tok=load(str(a.model));mx.set_cache_limit(96*1024**2)
    assert all(l.is_linear for l in model.model.layers[:3])
    generic=tok.apply_chat_template([dict(role="user",content="What has changed in this dialogue?")],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    probe_ids=tok.encode(generic,add_special_tokens=False);start=time.perf_counter()
    probe=first_layers(model,make_prompt_cache(model),probe_ids)["probe"];mx.synchronize()
    setup=time.perf_counter()-start;initial=time.monotonic()
    cases=json.loads((HERE/"raw_inputs.json").read_text());result={}
    for k,case in cases.items():
        mx.synchronize();tick=time.perf_counter()
        c=make_prompt_cache(model)
        ids=tok.encode(case["prior"],add_special_tokens=False)[-128:]
        first_layers(model,c,ids)
        scores=[]
        for turn in case["turns"]:
            ids=tok.encode(turn["text"],add_special_tokens=False);assert len(ids)==turn["tokens"]
            v=np.asarray(first_layers(model,c,ids,probe)["scores"]).tolist()
            assert all(np.isfinite(v))
            scores.append(v)
        mx.synchronize()
        result[k]=dict(scores=scores,seconds=time.perf_counter()-tick)
        print(k,round(time.monotonic()-initial,1),flush=True)
        if time.monotonic()-initial>900:raise TimeoutError("shallow screen")
    (HERE/"shallow_scores.json").write_text(json.dumps(dict(probe_setup_s=setup,probe_tokens=len(probe_ids),sessions=result))+"\n")


if __name__=="__main__":main()
