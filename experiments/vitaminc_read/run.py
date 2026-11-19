"""Collect label logits and layer-16 reads for matched evidence versions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask,create_ssm_mask
import numpy as np
from data import split,manifest

HERE=Path(__file__).resolve().parent
INSTRUCTION=("Use only the passage to judge the claim. Answer A if the passage supports it, B if it refutes it, "
             "or C if the passage does not provide enough information. Return only the letter.")


def tokens(tok,passage,claim):
    body=f"{INSTRUCTION}\n\nPassage: {passage}\n\nClaim: {claim}\nVerdict:"
    return tok.apply_chat_template([{"role":"user","content":body}],add_generation_prompt=True,enable_thinking=False,tokenize=True)


def project(model,hidden):
    lm=getattr(model,"language_model",model)
    if lm.args.tie_word_embeddings:return lm.model.embed_tokens.as_linear(hidden)
    return lm.lm_head(hidden)


def read(model,ids,labels,check):
    lm=getattr(model,"language_model",model);net=lm.model
    x=mx.array([ids]);h=net.embed_tokens(x)
    masks=create_attention_mask(h,None),create_ssm_mask(h,None)
    for i,layer in enumerate(net.layers,1):
        h=layer(h,mask=masks[int(layer.is_linear)],cache=None)
        if i==16:
            shallow=net.norm(h[:,-1,:])
            lens=project(model,shallow)[:,labels]
    deep=project(model,net.norm(h[:,-1,:]))
    y=deep[:,labels]
    mx.eval(shallow,lens,y)
    if check:
        reference=model(x)[:,-1,labels]
        # Different output sequence dimensions can change quantized kernel tiling.
        assert float(mx.max(mx.abs(reference.astype(mx.float32)-y.astype(mx.float32))).item())<=0.25
        assert int(reference.argmax().item())==int(y.argmax().item())
    return np.array(shallow.astype(mx.float32))[0],np.array(lens.astype(mx.float32))[0],np.array(y.astype(mx.float32))[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("data",type=Path);ap.add_argument("model",type=Path)
    ap.add_argument("cache",type=Path);ap.add_argument("part",choices=("fit","pilot","confirmation"))
    ap.add_argument("pins",type=Path);a=ap.parse_args()
    parts=split(a.data)
    assert manifest(parts)==json.loads((HERE/"split.json").read_text())
    for name,value in json.loads(a.pins.read_text())["model_files"].items():
        with (a.model/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==value
    model,tok=load(str(a.model));mx.set_cache_limit(96*1024**2)
    labels=[tok.encode(c,add_special_tokens=False) for c in "ABC"]
    assert all(len(x)==1 for x in labels)
    labels=mx.array([x[0] for x in labels])
    output=a.cache/a.part;output.mkdir(exist_ok=True,parents=True)
    start=time.monotonic()
    for ix,r in enumerate(parts[a.part]):
        name=output/(r["id"]+".npz")
        if name.exists():continue
        hs=[];ls=[];ys=[];counts=[]
        for j,claim in enumerate(r["claims"]):
            for k,passage in enumerate(r["passages"]):
                ids=tokens(tok,passage,claim)
                h,l,y=read(model,ids,labels,ix==0 and j==0 and k==0)
                hs.append(h);ls.append(l);ys.append(y);counts.append(len(ids))
        with name.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f,h=np.stack(hs).reshape(2,2,-1),lens=np.stack(ls).reshape(2,2,3),
                                full=np.stack(ys).reshape(2,2,3),gold=r["gold"],tokens=np.array(counts).reshape(2,2))
        name.with_suffix(".tmp").replace(name)
        mx.clear_cache()
        if ix%8==7:print(a.part,ix+1,round(time.monotonic()-start,1),flush=True)
        if time.monotonic()-start>1200:raise TimeoutError("split collection")
if __name__=="__main__":main()
