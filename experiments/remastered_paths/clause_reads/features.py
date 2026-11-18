"""Compute per-token contributions to the last native query by reverse propagation."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import sys
import time

import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gated_delta import compute_g

P=Path(__file__).resolve().parent
R=P.parent
def orth(rng,d,k):return mx.array(np.linalg.qr(rng.normal(size=(d,k)))[0].astype(np.float32))
def unmx(a):return np.asarray(a.astype(mx.float32))
def read(f):return json.loads(f.read_text())

def lex(word):
    grams={'word='+word}
    s='^'+word+'$'
    grams.update(s[i:i+j] for j in (2,3,4) for i in range(len(s)-j+1))
    v=np.zeros(160,dtype=np.float32)
    for g in grams:
        digest=int.from_bytes(hashlib.sha256(g.encode()).digest()[:4],'big')
        v[digest%len(v)]+=1 if digest//len(v)%2 else -1
    return v/max(np.linalg.norm(v),1e-8)

def prepare(tok,item):
    text='Question: '+item['text']+'?'
    encoding=tok.backend_tokenizer.encode(text,add_special_tokens=False)
    tokens=list(re.finditer(r'\w+',text))[1:]
    anchor=next((j for j,t in enumerate(tokens) if t[0]=='entity'),len(tokens)//2)
    # The entity is followed outward from its mention. Equidistant words to its right come first.
    seq=sorted(range(len(tokens)),key=lambda j:(abs(j-anchor),j<anchor,j))
    words=[tokens[j][0] for j in seq]
    spans=[[i for i,(a,b) in enumerate(encoding.offsets) if a<t.end() and b>t.start()] for t in [tokens[j] for j in seq]]
    assert all(spans)
    ps=np.array([[(j-anchor)/max(len(tokens),1),abs(j-anchor)/max(len(tokens),1),j/max(len(tokens),1),float(j<anchor)] for j in seq])
    return encoding.ids,words,spans,ps

def main():
    reg=read(P/'contract.json')
    for f,h in reg['pins'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h,f
    engine,tok=load(reg['model']);cfg=engine.language_model.args;items=read(R/'state_parser/items.json')
    grouped=defaultdict(list);max_words=0
    for j,row in enumerate(items):
        ids,words,spans,ps=prepare(tok,row);max_words=max(max_words,len(words))
        grouped[len(ids)].append((j,ids,words,spans,ps))
    rng=np.random.default_rng(61614);left=orth(rng,cfg.linear_value_head_dim,8)
    residual_proj=orth(rng,cfg.hidden_size,128)
    native_proj=orth(rng,cfg.linear_num_value_heads*8,128)
    lengths=np.zeros(len(items),dtype=np.int32);store={};kernel=qwen.gated_delta_update
    tick=time.monotonic();max_err=0.;nt=0
    for length,group in sorted(grouped.items()):
        for a in range(0,len(group),12):
            batch=group[a:a+12];B=len(batch);nt+=B*length
            hh=engine.model.embed_tokens(mx.array([r[1] for r in batch]));mask=create_attention_mask(hh,None);data={}
            def capture(q,k,v,x,b,al,bias,state,m,**kw):
                out,st=kernel(q,k,v,x,b,al,bias,state,m,**kw)
                data.update(q=q,k=k,v=v,b=mx.sigmoid(b),g=compute_g(al,x,bias),out=out)
                return out,st
            for level,l in enumerate(engine.model.layers[:7]):
                if level==6:qwen.gated_delta_update=capture
                try:hh=l(hh,mask=None if l.is_linear else mask,cache=None)
                finally:qwen.gated_delta_update=kernel
            projected=data['v'].astype(mx.float32)@left
            out=data['out'].astype(mx.float32)@left
            rep=cfg.linear_num_value_heads//cfg.linear_num_key_heads
            k=mx.repeat(data['k'].astype(mx.float32),rep,axis=2)
            w=mx.repeat(data['q'][:,-1].astype(mx.float32),rep,axis=1);endpoint=w
            weight=[];dots=[]
            retention=mx.ones((B,cfg.linear_num_value_heads),dtype=mx.float32)
            for i in reversed(range(length)):
                key=k[:,i];beta=data['b'][:,i].astype(mx.float32)
                inner=mx.sum(key*w,axis=-1)
                weight.append(retention*beta*inner)
                dots.append(beta*mx.sum(key*endpoint,axis=-1))
                w=w-(beta*inner)[...,None]*key
                retention*=data['g'][:,i].astype(mx.float32)
            weight=mx.stack(weight[::-1],axis=1);dots=mx.stack(dots[::-1],axis=1)
            approx=mx.sum(projected*weight[...,None],axis=1)
            err=float((mx.linalg.norm(approx-out[:,-1])/mx.maximum(mx.linalg.norm(out[:,-1]),1e-8)).item())
            max_err=max(max_err,err)
            def weighted(a):
                scaled=a*mx.rsqrt(mx.mean(a**2,axis=1,keepdims=True)+1e-12)
                return projected*mx.clip(scaled,-5,5)[...,None]
            def compress(x):return x.reshape(B,length,-1)@native_proj
            raw={n:unmx(v) for n,v in dict(resid=hh.astype(mx.float32)@residual_proj,write=compress(projected),read=compress(out),dot=compress(weighted(dots)),transport=compress(weighted(weight))).items()}
            for b,(ix,ids,words,spans,ps) in enumerate(batch):
                L=len(words);lengths[ix]=L
                features={n:np.stack([data[b,loc,:].mean(0) for loc in spans]) for n,data in raw.items()}
                features['lex']=np.stack([lex(word) for word in words]);features['pos']=ps
                for n,arr in features.items():
                    if n not in store:store[n]=np.zeros((len(items),max_words,arr.shape[-1]),dtype=np.float16)
                    store[n][ix,:L]=arr
    assert max_err<.04 and all(np.isfinite(v).all() for v in store.values()),max_err
    np.savez(P/'features.npz',lengths=lengths,**store)
    (P/'features_receipt.json').write_text(json.dumps(dict(n=len(items),positions=nt,wall_s=time.monotonic()-tick,max_relative_endpoint_reconstruction_error=max_err))+'\n')
    print('features',len(items),max_err,flush=True)
if __name__=='__main__':main()
