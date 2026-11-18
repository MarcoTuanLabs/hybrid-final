"""Compare retrospective insertion strategies on genuine, future-unseen user questions."""
from pathlib import Path
import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter

import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import compute_g

HERE = Path(__file__).resolve().parent
KERNEL = qwen.gated_delta_update
PREFIX = 'Read the dated conversation. Answer the final question using the most recent applicable information. Give only the short answer.\n\nConversation:\n'

def read(name):
    return json.loads((HERE/name).read_text())

def sync(c):
    mx.eval(*[layer.state for layer in c])
    mx.synchronize()

def forward(model, ids, cache, trace=None):
    def record(q,k,v,a,b,al,bias,state,mask,**kw):
        trace.append((k,a,b,al,bias,v.shape[-1]))
        return KERNEL(q,k,v,a,b,al,bias,state,mask,**kw)
    if trace is not None:
        qwen.gated_delta_update=record
    try:
        if ids:
            model(mx.array([ids]), cache=cache)
        sync(cache)
        if trace is not None:
            mx.eval(*[arr for row in trace for arr in row[:5]])
    finally:
        qwen.gated_delta_update=KERNEL

def splice(model, old, boundary, new, trace, mode):
    result=copy.deepcopy(old)
    linear=0
    for i, layer in enumerate(model.model.layers):
        if layer.is_linear:
            if mode in ('gates','transport'):
                k,a,b,al,bias,vdim=trace[linear]
                diff=new[i][1]-boundary[i][1]
                if mode=='gates':
                    retention=mx.prod(compute_g(al,a,bias),axis=1)
                    recovered=diff*retention[:,:,None,None]
                else:
                    zero_read=mx.zeros_like(k)
                    zero_value=mx.zeros((*a.shape,vdim),dtype=k.dtype)
                    _,recovered=KERNEL(zero_read,k,zero_value,a,b,al,bias,diff,None,use_kernel=True)
                result[i][1]=old[i][1]+recovered
            linear+=1
        else:
            ok,ov=old[i].keys_and_values()
            nk,nv=new[i].keys_and_values()
            start=boundary[i].offset
            suffix=ok[:,:,start:]
            bsz,heads,n,dim=suffix.shape
            shift=new[i].offset-start
            if shift:
                suffix=layer.self_attn.rope(suffix.reshape(-1,1,1,dim), offset=shift).reshape(bsz,heads,n,dim)
            keys=mx.concatenate([nk,suffix],axis=2)
            vals=mx.concatenate([nv,ov[:,:,start:]],axis=2)
            result[i].state=(keys,vals,keys.shape[2])
    sync(result)
    return result

def retrieve(blocks,question):
    word=lambda t:re.findall(r'\w+',t.lower())
    terms=set(word(question)); docs=[Counter(word(b)) for b in blocks]
    mean=sum(map(lambda d:sum(d.values()),docs))/len(docs)
    def score(i):
        doc=docs[i];length=sum(doc.values())
        return sum(math.log(1+(len(docs)-df+.5)/(df+.5))*doc[w]*2.2/(doc[w]+1.2*(.25+.75*length/mean))
                   for w in terms if doc[w] for df in [sum(w in d for d in docs)])
    return [blocks[i] for i in sorted(sorted(range(len(docs)),key=lambda i:(-score(i),i))[:3])]

def main():
    cfg=read('contract.json');model_id=sys.argv[1]
    for file,sha in cfg['pins'].items():
        assert hashlib.sha256((HERE/file).read_bytes()).hexdigest()==sha,file
    model,tok=load(cfg['models'][model_id])
    enc=lambda s:tok.encode(s,add_special_tokens=False)
    header=tok.apply_chat_template([dict(role='user',content=PREFIX+'SPLITPOINT')],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    before,after=header.split('SPLITPOINT')
    open_ids=enc(before);close_ids=enc(after)
    end_tokens=set(tok.eos_token_ids)
    def answer(cache,qids):
        logits=model(mx.array([qids]),cache=cache)[0,-1];first=np.asarray(logits.astype(mx.float32))
        ids=[];steps=0
        while len(ids)<28:
            token=int(mx.argmax(logits).item())
            if token in end_tokens:break
            ids.append(token)
            if '\n' in tok.decode(ids) or len(ids)==28:break
            logits=model(mx.array([[token]]),cache=cache)[0,-1];steps+=1
        mx.synchronize()
        return tok.decode(ids).strip(),ids,first,len(qids)+steps
    path=HERE/(model_id+'_rows.jsonl');examples=read('inputs.json')
    if path.exists():
        seen={x['id'] for x in map(json.loads,path.read_text().splitlines())}
        for uid in seen:
            assert sum(x['id']==uid for x in map(json.loads,path.read_text().splitlines()))==len(cfg['methods'])
    else:seen=set()
    start=time.monotonic()
    with path.open('a') as out:
        for row in examples:
            if row['id'] in seen:continue
            prior=enc('\n'.join(row['prior'])+'\n');insert=enc('\n'.join(row['insert'])+'\n')
            later=enc('\n'.join(row['later']))[:cfg['suffix_tokens']]
            assert len(later)>cfg['warm_tokens']
            h=cfg['warm_tokens']; s=cfg['suffix_tokens']
            qids=enc('\n\nQuestion on '+row['date']+': '+row['question'])+close_ids
            begin=time.monotonic()
            prefix=make_prompt_cache(model);forward(model,open_ids+prior,prefix)
            prefix_s=time.monotonic()-begin
            begin=time.monotonic()
            boundary=copy.deepcopy(prefix);forward(model,later[:h],boundary)
            old=copy.deepcopy(boundary);trace=[];forward(model,later[h:],old,trace)
            old_s=time.monotonic()-begin
            records=[];all_logits={}
            modes=cfg['methods'].copy();random.Random(row['id']).shuffle(modes)
            for mode in modes:
                begin=time.monotonic();cached=True
                if mode in ('last_session','bm25'):
                    cached=False
                    if mode=='last_session':
                        input_ids=open_ids+insert+qids
                    else:
                        blocks=row['prior']+row['insert']+tok.decode(later).splitlines()
                        input_ids=open_ids+enc('\n'.join(retrieve(blocks,row['question'])))+qids
                    cur=make_prompt_cache(model);nt=0
                else:
                    input_ids=qids
                    updated=copy.deepcopy(prefix)
                    if mode=='full':
                        forward(model,insert+later,updated);nt=len(insert)+len(later)
                        cur=updated
                    else:
                        forward(model,insert+later[:h],updated)
                        nt=len(insert)+h
                        cur=splice(model,old,boundary,updated,trace,mode)
                ans,seq,logits,used=answer(cur,input_ids);time_s=time.monotonic()-begin
                all_logits[mode]=logits
                tape_bytes=sum(y.nbytes for t in trace for y in t[:3]) if mode=='transport' else 0
                records.append(dict(id=row['id'],mode=mode,text=ans,seq=seq,update_answer_s=time_s,old_prefix_s=prefix_s if cached else 0,
                    old_suffix_s=old_s if cached else 0,update_answer_forward=nt+used,old_forward=len(open_ids)+len(prior)+len(later) if cached else 0,
                    suffix=len(later),insert=len(insert),question=len(qids),tape_bytes=tape_bytes,transported_steps=(len(later)-h) if mode=='transport' else 0))
                del cur
            ref=all_logits['full']
            for x in records:
                diff=all_logits[x['mode']]-ref;x['centered_logit_rmse']=float(np.sqrt(np.mean((diff-diff.mean())**2)))
                out.write(json.dumps(x,ensure_ascii=False)+'\n')
            out.flush()
            print(row['id'],len(prior),len(insert),len(later),round(time.monotonic()-start,1),flush=True)
            del prefix,boundary,trace,old
            mx.clear_cache()
    (HERE/(model_id+'_receipt.json')).write_text(json.dumps(dict(wall_s=time.monotonic()-start,n=len(examples),model=model_id,old_cost='built before question; standard lifecycle costs from common cache ingest, modes last_session/bm25 store only text',skipped=len(seen)))+'\n')

if __name__=='__main__':
    main()
