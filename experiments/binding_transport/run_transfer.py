"""Estimate paired changes to local attention storage without seeing a revised prompt at inference."""
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
from mlx_lm.models.cache import KVCache
from scipy.linalg import cho_factor,cho_solve,eigh

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"binding_cache"),str(HERE.parent/"binding_transfer")]
sys.path.insert(0,str(HERE.parent/"halumem_bridge"))
from cache_rotary import inverse_rotary
from docs import document
from inputs import load_inputs
# Load the cache-localization module without depending on Python's generic 'run' name.
import importlib.util
spec=importlib.util.spec_from_file_location("binding_cache_runner",HERE.parent/"binding_cache/run.py")
cache_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache_module)
prefill,oracle_cache=cache_module.prefill,cache_module.edited_cache
FORMATS=("DEFAULT","WORK")
MODES=("stale","full","notice","swap_values","centered","project","paired","local","recompute")


def role_positions(data,tok,seed,doc):
    rng=np.random.default_rng(seed);rng.permutation(len(data[0]))
    countries=[data[1][i][0] for i in rng.permutation(len(data[1]))[:3]]
    positions=[]
    for name in countries:
        token=tok.encode(" "+name,add_special_tokens=False)[0]
        ix=[i for i,v in enumerate(doc["old"]) if v==token]
        assert len(ix)==1,(name,ix)
        positions.append(ix[0])
    assert sorted(positions)==positions
    return positions


def rows_at(model,cache,pos):
    layers=[]
    for l,c in zip(model.model.layers,cache):
        if isinstance(c,KVCache):
            k=c.keys[:,:,pos:pos+2].astype(mx.float32)
            v=c.values[:,:,pos:pos+2].astype(mx.float32)
            rope=l.self_attn.rope
            k=inverse_rotary(rope,k,pos)
            layers.append(mx.stack([k[0].transpose(1,0,2),v[0].transpose(1,0,2)]))
    return np.asarray(mx.stack(layers))


def replace(model,cache,edits):
    result=[]
    for i,(layer,c) in enumerate(zip(model.model.layers,cache)):
        if not isinstance(c,KVCache):
            result.append(c);continue
        linear=i//4
        keys=c.keys[:,:,:c.offset];values=c.values[:,:,:c.offset]
        for pos,row in sorted(edits.items()):
            k,v=mx.array(row[linear]).transpose(0,2,1,3)
            k=layer.self_attn.rope(k[None],offset=pos).astype(keys.dtype)
            v=v[None].astype(values.dtype)
            keys=mx.concatenate((keys[:,:,:pos],k,keys[:,:,pos+2:]),axis=2)
            values=mx.concatenate((values[:,:,:pos],v,values[:,:,pos+2:]),axis=2)
        rep=KVCache();rep.state=(keys,values,c.offset);result.append(rep)
    return result


def calibrate(model,tok,data):
    start=time.perf_counter()
    source=[];target=[];deltas=[];roles=[]
    means=np.zeros((2,3,8,2,2,4,256),np.float64)
    for index,seed in enumerate(range(64932000,64932096)):
        for style,fmt in enumerate(FORMATS):
            doc=document(data,tok,seed,fmt);p=role_positions(data,tok,seed,doc)
            old=prefill(model,doc["old"]);fresh=prefill(model,doc["new"])
            old_rows=[rows_at(model,old,x) for x in p]
            means[style]+=old_rows
            for t,s in ((seed%3,(seed+1)%3),((seed+1)%3,seed%3)):
                source.append(old_rows[s]);target.append(old_rows[t])
                deltas.append(rows_at(model,fresh,p[t])-old_rows[t]);roles.append((style,s,t))
        if index%16==15:print(seed,round(time.perf_counter()-start,1),flush=True)
        if time.perf_counter()-start>720:raise TimeoutError("calibration")
    means=(means/96).astype(np.float32)
    x=np.stack([a-b-means[f,s]+means[f,t] for a,b,(f,s,t) in zip(source,target,roles)])
    y=np.stack(deltas);del source,target,deltas
    shape=x.shape[1:-1];assert shape==(8,2,2,4)
    bases=np.empty(shape+(256,32),np.float32);maps=np.empty_like(bases)
    for key in np.ndindex(shape):
        a=x[(slice(None),)+key].astype(np.float64);b=y[(slice(None),)+key].astype(np.float64)
        _,v=eigh(b.T@b,subset_by_index=(224,255),check_finite=False)
        gram=a.T@a
        gram.flat[::257]+=max(1e-12,.02*np.trace(gram)/256)
        bases[key]=v
        maps[key]=cho_solve(cho_factor(gram,lower=True,check_finite=False),a.T@(b@v),check_finite=False)
    np.savez(HERE/"fit.npz",means=means,bases=bases,maps=maps)
    rec=dict(documents=192,paired_row_examples=len(x),seconds=time.perf_counter()-start,
             parameter_bytes=means.nbytes+bases.nbytes+maps.nbytes,sha256=hashlib.sha256((HERE/"fit.npz").read_bytes()).hexdigest())
    (HERE/"fit.json").write_text(json.dumps(rec,indent=2)+"\n")


def recompute(model,doc,old):
    edits={}
    for pos in doc["changed"]:
        # Materialize the exact old prefix on demand for this quality comparator.
        # The timing comparison must instead retain these checkpoints on original ingest.
        prior=prefill(model,doc["old"][:pos])
        newer=prefill(model,doc["new"][pos:pos+2],prior)
        edits[pos]=rows_at(model,newer,pos)
    return replace(model,old,edits)


def evaluate(model,tok,data):
    rec=json.loads((HERE/"fit.json").read_text())
    assert hashlib.sha256((HERE/"fit.npz").read_bytes()).hexdigest()==rec["sha256"]
    with np.load(HERE/"fit.npz") as z:means,bases,maps=[z[k] for k in ("means","bases","maps")]
    out=HERE/"dev_rows.jsonl"
    seen={tuple(json.loads(l)["key"]) for l in out.read_text().splitlines()} if out.exists() else set()
    start=time.perf_counter()
    with out.open("a") as f:
        for seed in range(51973000,51973012):
            for style,fmt in enumerate(FORMATS):
                if (seed,fmt) in seen:continue
                doc=document(data,tok,seed,fmt);p=role_positions(data,tok,seed,doc)
                old=prefill(model,doc["old"]);fresh=prefill(model,doc["new"]);results={}
                t,s=seed%3,(seed+1)%3
                old_values={role:rows_at(model,old,p[role]) for role in (t,s)}
                for mode in MODES:
                    if mode=="notice":cache=prefill(model,doc["notice"],copy.deepcopy(old))
                    elif mode=="recompute":cache=recompute(model,doc,old)
                    elif mode in ("centered","project","paired"):
                        edits={}
                        for dest,src in ((t,s),(s,t)):
                            delta=old_values[src]-old_values[dest]-means[style,src]+means[style,dest]
                            if mode!="centered":
                                coeff=np.einsum("...i,...ir->...r",delta,bases if mode=="project" else maps)
                                delta=np.einsum("...ir,...r->...i",bases,coeff)
                            edits[p[dest]]=old_values[dest]+delta
                        cache=replace(model,old,edits)
                    else:cache=oracle_cache(old,fresh,doc["changed"],mode)
                    preds=[]
                    for q in doc["questions"]:
                        logits=model(mx.array([q]),cache=copy.deepcopy(cache))[0,-1]
                        preds.append(int(mx.argmax(logits).item()))
                    results[mode]=preds
                row={k:doc[k] for k in ("digest","stable","old_gold","new_gold")}
                row.update(key=[seed,fmt],outputs=results)
                f.write(json.dumps(row)+"\n");f.flush()
            print(seed,round(time.perf_counter()-start,1),flush=True)
            if time.perf_counter()-start>720:raise TimeoutError("development")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("phase",choices=("calibrate","dev"))
    for name in ("data","model","public"):p.add_argument(name,type=Path)
    a=p.parse_args()
    pins=json.loads((a.public/"experiments/read_projection/release.json").read_text())
    for k,h in pins["model_files"].items():
        with (a.model/k).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==h
    model,tok=load(str(a.model));mx.set_cache_limit(96*1024**2)
    data=load_inputs(a.data,tok)
    if a.phase=="calibrate":calibrate(model,tok,data)
    else:evaluate(model,tok,data)


if __name__=="__main__":main()
