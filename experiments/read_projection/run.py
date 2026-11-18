import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
import numpy as np
from codec import encode_delta,observability

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import argparse

p=Path(__file__).resolve().parent
read=lambda f: json.loads(f.read_text())

class Reader:
    def __init__(self,path):
        self.model,self.tok=load(str(path))
        self.kernel=qwen.gated_delta_update
    def prefill(self,ids):
        cache=make_prompt_cache(self.model)
        self.model(mx.array([ids]),cache=cache)
        mx.eval(*[layer.state for layer in cache])
        return cache

def encode(trial,facts,question):
    body='Read the story in order. Answer the question with the location only.\n\n'+'\n'.join(facts)+'\n\n'+question
    text=trial.tok.apply_chat_template([dict(role='user',content=body)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=trial.tok.encode(text,add_special_tokens=False)
    op=trial.tok.encode('<think>',add_special_tokens=False)
    assert len(op)==1 and ids.count(op[0])==1
    cut=ids.index(op[0]);assert len(ids)-cut==4
    return ids[:cut],ids[cut:]

def encode_mquake(trial,facts,q,instruction):
    body=instruction+'\n\nFacts:\n'+'\n'.join(facts)+'\n\nQuestion: '+q+'\nAnswer:'
    text=trial.tok.apply_chat_template([dict(role='user',content=body)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=trial.tok.encode(text,add_special_tokens=False);op=trial.tok.encode('<think>',add_special_tokens=False)
    cut=ids.index(op[0]);assert len(ids)-cut==4
    return ids[:cut],ids[cut:]

def host(x):
    return np.ascontiguousarray(np.asarray(x.astype(mx.float32)))

def run(trial,cache,tail,limit,collect=False):
    c=copy.deepcopy(cache);kernel=trial.kernel;captured=[];logits_at=[]
    for t in tail:
        slots=[]
        def observe(q,k,v,a,b,al,bias,state,mask,**kw):
            slots.append((q[0,0],k[0,0],mx.sigmoid(b)[0,0]))
            return kernel(q,k,v,a,b,al,bias,state,mask,**kw)
        if collect:qwen.gated_delta_update=observe
        try:
            logits=trial.model(mx.array([[t]]),cache=c)[0,-1];mx.eval(logits)
            logits_at.append(host(logits))
            if collect:captured.append([tuple(host(x) for x in entry) for entry in slots])
        finally:qwen.gated_delta_update=kernel
    tokens=[]
    for _ in range(limit):
        t=int(mx.argmax(logits).item())
        if t in trial.tok.eos_token_ids:break
        tokens.append(t)
        if '\n' in trial.tok.decode(tokens):break
        logits=trial.model(mx.array([[t]]),cache=c)[0,-1]
    return logits_at,tokens,captured

def build(trial,old,new,captured,mode):
    c=copy.deepcopy(new)
    nbytes=0;dense_bytes=0;diagnostic=[]
    recurrent=[i for i,l in enumerate(trial.model.model.layers) if l.is_linear]
    assert all(len(t)==len(recurrent) for t in captured)
    for j,i in enumerate(recurrent):
        old_state=old[i][1];new_state=new[i][1]
        assert old_state.dtype==new_state.dtype==mx.float32
        a=host(old_state[0]);b=host(new_state[0]);delta=b-a
        dense_bytes+=delta.nbytes
        if mode=='dense':
            nbytes+=delta.nbytes;continue
        if mode=='no_matrix':
            c[i][1]=old_state;continue
        q,w=observability([t[j] for t in captured],len(a))
        recovered,n=encode_delta(delta,mode,q,w);nbytes+=n
        c[i][1]=mx.array((a+recovered)[None],dtype=old_state.dtype)
        true=delta@w;error=(delta-recovered)@w
        diagnostic.append(float(np.linalg.norm(error)/max(np.linalg.norm(true),1e-12)))
    return c,nbytes,dense_bytes,diagnostic

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['dev','test'])
    parser.add_argument('model_path',type=Path)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args()
    phase=args.phase; out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    cfg=read(p/'release.json')
    for f,h in cfg['model_files'].items():
        with (args.model_path/f).open('rb') as fp:
            assert hashlib.file_digest(fp,'sha256').hexdigest()==h,f
    assets=p/'data'/(phase+'_inputs.json')
    assert hashlib.sha256(assets.read_bytes()).hexdigest()==cfg['inputs'][phase]
    inputs=read(assets);instruction=cfg['mquake_instruction']
    trial=Reader(args.model_path);start=time.perf_counter();rng=random.Random(3527)
    rng.shuffle(inputs)
    with (out/(phase+'_rows.jsonl')).open('x') as log:
        for ii,x in enumerate(inputs):
            src=x['source'];ident=x['id'];limit=12 if src=='babi' else 24
            for role,key in [('changed','q'),('stable','q_control')]:
                def ids(facts):
                    return encode(trial,facts,x[key]) if src=='babi' else encode_mquake(trial,facts,x[key],instruction)
                a,tail=ids(x['old']);b,tail2=ids(x['new']);assert tail==tail2
                old=trial.prefill(a);new=trial.prefill(b)
                gold_logits,tokens,trace=run(trial,new,tail,limit,True)
                mode_order=cfg['modes'].copy();rng.shuffle(mode_order)
                for mode in mode_order:
                    t=time.perf_counter();c,n,dense,errors=build(trial,old,new,trace,mode)
                    codec_s=time.perf_counter()-t
                    logits,seq,_=run(trial,c,tail,limit)
                    diff=[a.astype(float)-b for a,b in zip(logits,gold_logits)]
                    if mode=='dense':assert all(np.all(z==0) for z in diff) and tokens==seq
                    rmse=np.sqrt(np.mean([np.mean((z-z.mean())**2) for z in diff]))
                    row=dict(source=src,id=ident,role=role,mode=mode,bytes=n,dense_bytes=dense,text=trial.tok.decode(seq),tokens=seq,
                        same_tokens=seq==tokens,first_match=bool(logits[-1].argmax()==gold_logits[-1].argmax()),rmse=float(rmse),
                        max_abs_logit_error=max(float(np.max(np.abs(z))) for z in diff),codec_s=codec_s,
                        median_read_error=None if not errors else float(np.median(errors)),max_read_error=None if not errors else max(errors))
                    print(json.dumps(row),file=log)
                log.flush()
            print('cases',ii+1,len(inputs),round(time.perf_counter()-start,1),flush=True)
    (out/(phase+'_receipt.json')).write_text(json.dumps(dict(wall_s=time.perf_counter()-start,revision=cfg['revision']))+'\n')

if __name__=='__main__':main()
