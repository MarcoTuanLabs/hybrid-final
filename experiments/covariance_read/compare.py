"""Compare prefix-key covariance and a finite post-normalization read contrast on one write."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE=Path(__file__).resolve().parent


def norm_rows(x):
    return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)


def unit(v):
    return v/max(np.linalg.norm(v),1e-30)


def whiten(cov,key,n):
    ridge=max(1e-12,1e-3*np.trace(cov)/len(n))
    v=cho_solve(cho_factor(cov+ridge*np.eye(len(n)),lower=True,check_finite=False),key,check_finite=False)
    return unit(v)*np.linalg.norm(n)*(-1 if v@n<0 else 1)


def prefill_keys(trial,ids):
    stored=[]
    def hook(q,k,v,*args,**kw):
        i=hook.count;hook.count+=1
        if i==trial.ordinal:stored.append(k[0,:,14,:])
        return trial.kernel(q,k,v,*args,**kw)
    hook.count=0
    qwen.gated_delta_update=hook
    try:cache=trial.prefill(ids)
    finally:qwen.gated_delta_update=trial.kernel
    assert len(stored)==1 and stored[0].shape==(len(ids),128)
    return cache,np.asarray(stored[0].astype(mx.float32),dtype=np.float64)


def plans(trial,hist,original,edited):
    names=json.loads((HERE/"contract.json").read_text())["policies"]
    choices={m:[] for m in names};stats=[]
    factor=None
    for left,right in zip(original,edited,strict=True):
        for key in ("key","beta","n","gate"):assert np.array_equal(left[key],right[key])
        n=left["n"];s=left["s"];d=np.ascontiguousarray(right["s"]-s);k=left["key"]
        # Track the original state-Gram reader's first transported source-key column.
        if factor is not None:
            factor=left["gate"][0]*(factor-left["beta"][0]*k*(k@factor))
        elif np.any(left["v"][0]!=right["v"][0]):
            factor=k.copy()
        source=n if factor is None else factor
        hist=np.vstack([hist,k])
        x=hist/np.maximum(np.linalg.norm(hist,axis=1,keepdims=True),1e-30)
        centered=x-x.mean(axis=0);cov=(centered.T@centered)/len(x)
        # q is in kernel coordinates; the comparison holds its norm fixed.
        curve=cov@n
        candidates=[np.linalg.norm(n)*unit(n-alpha*curve) for alpha in (0,.25,.5,.75,1)]
        def diff(q):return norm_rows(q@(s+d))-norm_rows(q@s)
        target=unit(diff(n))
        values=[float(np.sum(diff(q)*target)) for q in candidates]
        best=int(np.argmax(values))
        opt=trial.solve((s,d,n,[]),"ref16")
        sol=dict(native=n,ccq_half=candidates[2],ccq_one=candidates[4],ccq_write=candidates[best],
                 key_cov_white=whiten(cov,source,n),state_white=whiten(sum(m@m.T for m in s),source,n),normalized=opt)
        angles={}
        for m,q in sol.items():
            cos=float(unit(q)@unit(n))
            assert abs(np.linalg.norm(q)/np.linalg.norm(n)-1)<1e-7
            if m not in ("key_cov_white","state_white"):assert cos>=.5-1e-7
            choices[m].append(q);angles[m]=float(np.degrees(np.arccos(np.clip(cos,-1.,1.))))
        stats.append(dict(history=len(hist),chosen_lambda=best/4,angles=angles))
    return choices,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=["dev","test"]);p.add_argument("model",type=Path);p.add_argument("public",type=Path)
    a=p.parse_args();repo=a.public.resolve()
    sys.path.insert(0,str(repo/"experiments/quantized_write"))
    from run import Trial,make_example
    if a.phase=="test":assert json.loads((HERE/"dev_summary.json").read_text())["development_pass"]
    cfg=json.loads((repo/"evidence/quantized_9b/contract.json").read_text())
    pins=json.loads((repo/"experiments/read_projection/release.json").read_text())
    for name,sha in pins["model_files"].items():
        with (a.model/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==sha
    trial=Trial(a.model,cfg);mx.set_cache_limit(96*1024**2)
    cases=range(27112000,27112016) if a.phase=="dev" else range(72984000,72984032)
    modes=json.loads((HERE/"contract.json").read_text())["policies"]
    dst=HERE/(a.phase+"_rows.jsonl")
    seen={tuple(json.loads(line)["key"]) for line in dst.read_text().splitlines()} if dst.exists() else set()
    start=time.perf_counter()
    with dst.open("a") as out:
        for seed in cases:
            for surface in ("table","prose"):
                for nuisance in (False,True):
                    key=(seed,surface,nuisance)
                    if key in seen:continue
                    ex=make_example(*key);aid,tail=trial.encode(ex,"a");bid,tail_b=trial.encode(ex,"b")
                    assert tail==tail_b and len(tail)==4
                    base,hist=prefill_keys(trial,aid)
                    donor=trial.prefill(bid);edit=trial.write(base,donor)
                    ids=[trial.tok.encode(ex[k],add_special_tokens=False)[0] for k in ("value_a","value_b")]
                    ref0,b=trial.steps(base,tail,capture=True);ref1,e=trial.steps(edit,tail,capture=True)
                    qs,stats=plans(trial,hist,b,e)
                    order=modes.copy();random.Random(754*seed+3*nuisance+5*(surface=="prose")).shuffle(order)
                    result={}
                    for m in order:
                        top=[];margin=[]
                        for state,ref in ((base,ref0),(edit,ref1)):
                            logits,_=trial.steps(state,tail,qs[m])
                            if m=="native":assert np.array_equal(logits,ref)
                            top.append(int(np.argmax(logits)))
                            margin.append(float(logits[ids[1]]-logits[ids[0]]))
                        result[m]=dict(top=top,margin=margin)
                    out.write(json.dumps(dict(key=key,recipient=ids[0],donor=ids[1],methods=result,slots=stats))+"\n");out.flush()
            print(seed,round(time.perf_counter()-start,1),flush=True);mx.clear_cache()
            if time.perf_counter()-start>1080:raise TimeoutError("phase clock")


if __name__=="__main__":
    main()
