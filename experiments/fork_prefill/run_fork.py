"""Compare both packed schedules against the batched and suffix-fused 9B readers."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import mlx.core as mx
import numpy as np
from packed import HERE,forward
from service import validate,execute

def request(t,ex,m,active,deps):
    if m in ('joint','window'):
        row,y,slot=execute(t,ex,m,active,deps)
        return row,y,slot
    mx.synchronize();begin=time.perf_counter()
    a,tail=t.encode(ex,'a');b,second=t.encode(ex,'b');assert tail==second
    y,q,slot,orig,n=forward(t,a,b,tail,active,m,deps)
    top=int(y.argmax());mx.synchronize();wall=time.perf_counter()-begin
    slot=np.asarray(slot.astype(mx.float32))
    assert np.isfinite(y).all() and len(q)==4
    return dict(top=top,seconds=wall,query_digest=hashlib.sha256(np.asarray(q).tobytes()).hexdigest(),
                write_digest=hashlib.sha256(slot.tobytes()).hexdigest(),original_digest=hashlib.sha256(np.asarray(orig.astype(mx.float32)).tobytes()).hexdigest()),y,slot

def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('repo',type=Path);p.add_argument('part',choices=['smoke','pilot','confirm']);p.add_argument('out',type=Path);a=p.parse_args()
    trial,deps=validate(a.model,a.repo);mx.set_cache_limit(96*1024**2)
    spec=json.loads((HERE/'contract.json').read_text());modes=spec['pilot_arms'] if a.part!='confirm' else spec['confirm_arms']
    start=time.monotonic();example=deps[1](27112000,'table',False)
    for m in modes:
        warm=request(trial,example,m,True,deps)
        if a.part=='smoke':print(m,warm[0]['top'],warm[0]['seconds'],flush=True)
    if a.part=='smoke':return
    a.out.mkdir(exist_ok=True,parents=True);dest=a.out/'rows.jsonl'
    seen={tuple(json.loads(l)['key']) for l in dest.read_text().splitlines()} if dest.exists() else set()
    with dest.open('a') as f:
        for seed in spec[a.part+'_seeds']:
            for surf in ('table','prose'):
                for nuisance in (False,True):
                    ex=deps[1](seed,surf,nuisance);aid,tail=trial.encode(ex,'a');bid,_=trial.encode(ex,'b')
                    common=next(i for i,(x,y) in enumerate(zip(aid,bid)) if x!=y)
                    gold=[trial.tok.encode(ex[k],add_special_tokens=False)[0] for k in ('value_a','value_b')]
                    for active in (True,False):
                        key=(seed,surf,nuisance,active)
                        if key in seen:continue
                        arms=list(modes);random.Random(seed*39+11*nuisance+37*active+73*(surf=='table')).shuffle(arms)
                        out={m:request(trial,ex,m,active,deps) for m in arms};results={}
                        for m,(row,y,w) in out.items():
                            results[m]=dict(row,margin=float(y[gold[1]]-y[gold[0]]),max_logit_vs_joint=float(np.max(np.abs(y-out['joint'][1]))),write_max_abs_vs_joint=float(np.max(np.abs(w-out['joint'][2]))))
                        f.write(json.dumps(dict(key=key,order=arms,recipient=gold[0],donor=gold[1],common=common,prefix=len(aid),suffix=len(tail),arms=results))+'\n');f.flush()
            mx.clear_cache();print(a.part,seed,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>spec['deadline_seconds'][a.part]:raise TimeoutError(a.part)
if __name__=='__main__':main()
