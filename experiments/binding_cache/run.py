"""Localize stored bindings before future CAPITALS questions arrive."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache,make_prompt_cache
from docs import document

HERE=Path(__file__).resolve().parent
MODES=("stale","full","recurrent","attention","country","local","values","keys","swap_values","notice","reassembled")


def prefill(model,tokens,c=None):
    if c is None:c=make_prompt_cache(model)
    model.model(mx.array([tokens]),cache=c)
    mx.eval(*[layer.state for layer in c])
    return c


def edited_cache(old,new,positions,mode):
    if mode=="stale":return old
    if mode=="full":return new
    state=[]
    for a,b in zip(old,new):
        if not isinstance(a,KVCache):
            state.append(b if mode in ("recurrent","reassembled") else a)
            continue
        if mode in ("attention","reassembled"):
            state.append(b);continue
        if mode=="recurrent":
            state.append(a);continue
        count=a.offset;assert b.offset==count
        p=positions if mode=="country" else sorted(positions+[i+1 for i in positions])
        to=mx.array(p);idx=mx.arange(count)
        if mode=="swap_values":
            src=(p[2:]+p[:2])
            mapping=list(range(count))
            for target,source in zip(p,src):mapping[target]=source
            key=a.keys[:,:,:count];val=mx.take(a.values[:,:,:count],mx.array(mapping),axis=2)
        else:
            mask=mx.any(idx[:,None]==to[None,:],axis=1)[None,None,:,None]
            def choose(x,y):return mx.where(mask,y[:,:,:count],x[:,:,:count])
            key=a.keys[:,:,:count] if mode=="values" else choose(a.keys,b.keys)
            val=a.values[:,:,:count] if mode=="keys" else choose(a.values,b.values)
        layer=KVCache();layer.state=(key,val,count);state.append(layer)
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("data","model","public"):p.add_argument(name,type=Path)
    a=p.parse_args();repo=a.public.resolve()
    source=HERE.parent/"binding_transfer"
    sys.path.insert(0,str(source))
    from inputs import load_inputs
    pins=json.loads((repo/"experiments/read_projection/release.json").read_text())
    for k,h in pins["model_files"].items():
        with (a.model/k).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==h
    model,tok=load(str(a.model));data=load_inputs(a.data,tok)
    mx.set_cache_limit(96*1024**2)
    out=HERE/"rows.jsonl"
    seen={tuple(json.loads(l)["key"]) for l in out.read_text().splitlines()} if out.exists() else set()
    start=time.monotonic()
    with out.open("a") as f:
        for seed in range(51973000,51973012):
            for fmt in ("DEFAULT","WORK"):
                if (seed,fmt) in seen:continue
                doc=document(data,tok,seed,fmt)
                old=prefill(model,doc["old"]);new=prefill(model,doc["new"])
                results={};digests={}
                for mode in MODES:
                    if mode=="notice":cache=prefill(model,doc["notice"],copy.deepcopy(old))
                    else:cache=edited_cache(old,new,doc["changed"],mode)
                    answers=[]
                    for i,q in enumerate(doc["questions"]):
                        logits=model(mx.array([q]),cache=copy.deepcopy(cache))[0,-1].astype(mx.float32)
                        data_logits=np.asarray(logits)
                        answers.append(dict(top=int(data_logits.argmax()),delta_margin=float(data_logits[doc["new_gold"][i]]-data_logits[doc["old_gold"][i]])))
                        if mode in ("full","reassembled"):
                            signature=hashlib.sha256(data_logits.tobytes()).hexdigest()
                            if mode=="full":digests[i]=signature
                            else:assert signature==digests[i]
                    results[mode]=answers
                row={k:doc[k] for k in ("old_gold","new_gold","stable","changed","digest")}
                row.update(key=[seed,fmt],outputs=results,old_tokens=len(doc["old"]),question_tokens=list(map(len,doc["questions"])),
                           replay_tokens=len(doc["new"])-min(doc["changed"]),notice_tokens=len(doc["notice"]))
                f.write(json.dumps(row)+"\n");f.flush()
            mx.clear_cache();print(seed,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>600:raise TimeoutError("localization")


if __name__=="__main__":main()
