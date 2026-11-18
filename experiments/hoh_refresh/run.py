"""Execute each two-edition policy separately and count the full old-to-new lifecycle."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
from mlx_lm.models import qwen3_next
from mlx_lm.models.cache import make_prompt_cache

HERE=Path(__file__).resolve().parent
SHARED=HERE.parent/'bridge_reads'
if not SHARED.is_dir():
    SHARED=HERE.parent/'remastered_paths'/'bridge_reads'
sys.path.insert(0,str(SHARED))
from engine import Engine
from repair import revise

def encode(engine,row,key):
    body=('Answer the question about this Wikipedia page. Give only the shortest factual answer.\n'
          +'Page: '+row['title']+'\nQuestion: '+row['question']+'\nEvidence:\n')
    text=engine.chat(body+row[key])
    start=text.index(body)+len(body)
    data=engine.tokenizer._tokenizer(text, add_special_tokens=False,return_offsets_mapping=True)
    ids=list(data.input_ids)
    assert ids==engine.encode(text)
    evidence={i:(max(0,x-start),min(len(row[key]),y-start)) for i,(x,y) in enumerate(data.offset_mapping)
              if x<start+len(row[key]) and y>start}
    return ids,evidence

@contextmanager
def capture(engine,cache,evidence,active,record):
    if not active:
        yield
        return
    targets={id(c) for layer,c in zip(engine.model.model.layers,cache,strict=True) if not layer.is_linear}
    last=[id(c) for c in cache if id(c) in targets][-2:]
    indices=list(evidence)
    original=qwen3_next.scaled_dot_product_attention
    weights=[]
    def wrapper(queries,keys,values,cache,scale,mask,**kwargs):
        if id(cache) in last:
            # Restrict to the evidence before normalizing, independently for each head.
            k=keys[:,:,indices,:].astype(mx.float32)
            n=queries.shape[1]//k.shape[1]
            q=queries[:,:,-1:,:].astype(mx.float32).reshape(1,k.shape[1],n,k.shape[-1])
            scores=(q@k.transpose(0,1,3,2))*scale
            weights.append(mx.mean(mx.softmax(scores,axis=-1),axis=(0,1,2)))
        return original(queries,keys,values,cache=cache,scale=scale,mask=mask,**kwargs)
    qwen3_next.scaled_dot_product_attention=wrapper
    try:
        yield
    finally:
        qwen3_next.scaled_dot_product_attention=original
    assert len(weights)==2 and indices
    att=(weights[0]+weights[1])*.5
    top=mx.argsort(-att)[:4].tolist()
    vals=att[mx.array(top)].tolist()
    record.extend([(*evidence[indices[i]],v) for i,v in zip(top,vals,strict=True)])

def decode(engine,ids,cache,start,evidence=None,save=False,trace=False):
    checkpoints={}
    stride=32 if save else 96
    for i in range(start,len(ids)-1,stride):
        end=min(i+stride,len(ids)-1)
        h=engine.model.model(mx.array([ids[i:end]]),cache=cache)
        mx.eval(h);engine.positions+=end-i
        if save:
            checkpoints[end]=copy.deepcopy(cache)
    output=[];records=[];tokens=ids[-1:]
    for _ in range(32):
        src=[]
        with capture(engine,cache,evidence,trace,src):
            logits=engine.forward(tokens,cache)
        token=int(mx.argmax(logits).item())
        if token in engine.tokenizer.eos_token_ids:break
        before=engine.tokenizer.decode(output)
        output.append(token);after=engine.tokenizer.decode(output)
        if trace:
            records.append(dict(answer=[len(before),len(after)],source=src))
        if '\n' in after: break
        tokens=[token]
    mx.synchronize()
    answer=engine.tokenizer.decode(output).split('\n',1)[0]
    return answer,records,checkpoints

def execute(engine,row,mode):
    tick=time.monotonic();positions=engine.positions
    old_ids,source=encode(engine,row,'old')
    retain=mode!='full'
    old,trace,checks=decode(engine,old_ids,make_prompt_cache(engine.model),0,source,retain,mode=='native')
    first=engine.positions-positions
    old_s=time.monotonic()-tick
    fresh=None;decision='replay'
    if mode in ('text','unique','native'):
        fresh,decision=revise(row['old'],row['current'],old,mode,trace)
    if fresh is None:
        new_ids,_=encode(engine,row,'current')
        shared=next((k for k,(a,b) in enumerate(zip(old_ids,new_ids)) if a!=b),min(len(old_ids),len(new_ids)))
        split=max((k for k in checks if k<=min(shared,len(new_ids)-1)),default=0)
        host=checks.get(split) if split else make_prompt_cache(engine.model)
        fresh,_,_=decode(engine,new_ids,host,split)
    mx.synchronize()
    return dict(old=old.strip(),current=fresh.strip(),decision=decision,old_positions=first,
                total_positions=engine.positions-positions,seconds=time.monotonic()-tick,old_s=old_s,
                trace_entries=sum(len(x['source']) for x in trace),checkpoints=len(checks))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    items=json.loads((HERE/'inputs.json').read_text());modes=['full','prefix','text','unique','native']
    path=HERE/'rows.jsonl'
    already={r['id'] for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    start=time.monotonic()
    with path.open('a') as out:
        for j,row in enumerate(items):
            if row['id'] in already:continue
            scores={}
            for mode in modes[j%5:]+modes[:j%5]:
                mx.clear_cache()
                scores[mode]=execute(engine,row,mode)
            out.write(json.dumps(dict(id=row['id'],policies=scores),ensure_ascii=False)+'\n');out.flush()
            print(j+1,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('registered thirty-minute budget')

if __name__=='__main__':
    main()
