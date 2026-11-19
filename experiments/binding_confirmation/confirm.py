"""Test the location of country updates and a structured, donor-free query baseline."""
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
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"binding_cache"),str(HERE.parent/"binding_transfer")]
from inputs import load_inputs
from docs import document
from run import edited_cache,prefill

MODES=("stale","full","recurrent","local","values","largest","outside","notice","redirect")
SEEDS=range(75219000,75219032)
FORMATS=("DEFAULT","WORK")


def split(data,tok):
    def digests(seeds):
        return {document(data,tok,s,f)["digest"] for s in seeds for f in FORMATS}
    prior=list(range(51973000,51973012))+list(range(51974000,51974064))+list(range(64932000,64932096))+list(range(64934000,64934048))
    new=digests(SEEDS);old=digests(prior)
    assert len(new)==2*len(SEEDS) and old.isdisjoint(new)
    return dict(test_documents=len(new),previous_or_reserved_documents=len(old),overlap=0,hashes=sorted(new))


def energies(old,new):
    total=None
    for a,b in zip(old,new):
        if isinstance(a,KVCache):
            n=a.offset
            assert b.offset==n
            for x,y in ((a.keys,b.keys),(a.values,b.values)):
                delta=x[:,:,:n].astype(mx.float32)-y[:,:,:n].astype(mx.float32)
                v=mx.sum(delta*delta,axis=(0,1,3))
                total=v if total is None else total+v
    energy=np.asarray(total)
    assert np.isfinite(energy).all()
    return np.argsort(-energy,kind="stable").tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=("split","run"))
    for arg in ("data","model","public"):parser.add_argument(arg,type=Path)
    a=parser.parse_args();data_path=a.data
    if a.action=="split":
        tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
        result=split(load_inputs(data_path,tok),tok)
        (HERE/"split.json").write_text(json.dumps(result,indent=2)+"\n")
        return
    known=json.loads((HERE/"split.json").read_text())["hashes"]
    pins=json.loads((a.public/"experiments/read_projection/release.json").read_text())
    for name,digest in pins["model_files"].items():
        with (a.model/name).open("rb") as file:assert hashlib.file_digest(file,"sha256").hexdigest()==digest
    model,tok=load(str(a.model));mx.set_cache_limit(96*1024**2)
    data=load_inputs(data_path,tok);out=HERE/"rows.jsonl"
    seen={tuple(json.loads(l)["key"]) for l in out.read_text().splitlines()} if out.exists() else set()
    start=time.monotonic()
    with out.open("a") as f:
        for seed in SEEDS:
            for fmt in FORMATS:
                if (seed,fmt) in seen:continue
                doc=document(data,tok,seed,fmt);assert doc["digest"] in known
                old=prefill(model,doc["old"]);new=prefill(model,doc["new"])
                sites=set(doc["changed"]+[j+1 for j in doc["changed"]])
                ordered=energies(old,new);best=sorted(ordered[:4]);outside=sorted([i for i in ordered if i not in sites][:4])
                mapping=list(range(3));left=seed%3;right=(seed+1)%3;mapping[left],mapping[right]=right,left
                answers={};native_digests=[]
                for m in MODES:
                    if m in ("largest","outside"):c=edited_cache(old,new,best if m=="largest" else outside,"country")
                    elif m=="notice":c=prefill(model,doc["notice"],copy.deepcopy(old))
                    elif m=="redirect":c=old
                    else:c=edited_cache(old,new,doc["changed"],m)
                    tops=[]
                    for i in range(3):
                        source=mapping[i] if m=="redirect" else i
                        logits=model(mx.array([doc["questions"][source]]),cache=copy.deepcopy(c))[0,-1].astype(mx.float32)
                        values=np.asarray(logits);tops.append(int(values.argmax()))
                        if m in ("stale","redirect"):
                            digest=hashlib.sha256(values.tobytes()).hexdigest()
                            if m=="stale":native_digests.append(digest)
                            else:assert digest==native_digests[source]
                    answers[m]=tops
                row={k:doc[k] for k in ("digest","stable","old_gold","new_gold")}
                row.update(key=[seed,fmt],local=sorted(sites),largest=best,outside=outside,outputs=answers,
                           redirection_logits_equal=True,document_tokens=len(doc["old"]))
                f.write(json.dumps(row)+"\n");f.flush()
            mx.clear_cache();print(seed,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>1200:raise TimeoutError("32-seed confirmation")


if __name__=="__main__":main()
