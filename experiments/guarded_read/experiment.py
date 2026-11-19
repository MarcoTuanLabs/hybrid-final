"""Test a background-anchored readout on real paired judgments with a matched-drift control."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
import numpy as np
from guard import compile_guard,solve,match,drift

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'write_window'))
from service import validate

def apply(trial,c,tail,plan):
    i=-1
    def hook(q,k,v,a,b,al,bias,state,mask,**kwargs):
        nonlocal i
        i+=1
        if i==trial.ordinal:
            override=mx.array(np.stack(plan),dtype=q.dtype)[None,:,None,:]
            q=mx.concatenate([q[:,:,:14],override,q[:,:,15:]],axis=2)
        return trial.kernel(q,k,v,a,b,al,bias,state,mask,**kwargs)
    qwen.gated_delta_update=hook
    try:
        y=trial.model(mx.array([tail]),cache=copy.deepcopy(c))[0,-1]
        return np.asarray(y.astype(mx.float32))
    finally:qwen.gated_delta_update=trial.kernel

def main():
    p=argparse.ArgumentParser()
    for n in ('model','repo','data','out','build'):p.add_argument(n,type=Path)
    a=p.parse_args();repo=a.repo.resolve();t,deps=validate(a.model,repo);stream=deps[5];set_write=deps[3]
    lib=compile_guard(repo,a.build)
    source=repo/'experiments/vitaminc_read';sys.path.append(str(source))
    from data import split,manifest
    spec=importlib.util.spec_from_file_location('prompt_tools',source/'run.py')
    prompts=importlib.util.module_from_spec(spec);spec.loader.exec_module(prompts)
    parts=split(a.data);assert manifest(parts)==json.loads((source/'split.json').read_text())
    raw=repo/'experiments/verdict_write/pilot/rows.jsonl'
    if not raw.exists():raw=HERE.parent/'verdict_write/pilot/rows.jsonl'
    original={z['id']:z for z in map(json.loads,raw.read_text().splitlines())}
    labels=t.tok.encode('A',add_special_tokens=False)+t.tok.encode('B',add_special_tokens=False)+t.tok.encode('C',add_special_tokens=False)
    assert len(labels)==3
    think=t.tok.encode('<think>',add_special_tokens=False)[0];initial=t.solve
    a.out.mkdir(exist_ok=True,parents=True);dest=a.out/'rows.jsonl'
    seen={json.loads(l)['id'] for l in dest.read_text().splitlines()} if dest.exists() else set()
    start=time.monotonic();mx.set_cache_limit(96*1024**2)
    for ix,r in enumerate(parts['pilot']):
        if r['id'] in seen:continue
        entries=[]
        for ci,(claim,gold) in enumerate(zip(r['claims'],r['gold'])):
            cs=[];tails=[]
            for text in r['passages']:
                ids=prompts.tokens(t.tok,text,claim);length=ids.index(think)
                assert len(ids)-length==4;cs.append(t.prefill(ids[:length]));tails.append(ids[length:])
            assert tails[0]==tails[1];tail=tails[0]
            for j in (0,1):
                old=cs[1-j];new=cs[j];w=new[28][0][:,-1:,t.lo:t.lo+256];edited=set_write(t,old,w)
                plans={k:[] for k in ('original','guard','matched')};geom=[]
                def design(slot,mode):
                    s,d,n,_=slot;q=initial(slot,mode);g=solve(lib,s,d,n,1.0);z=solve(lib,s,d,n,0.0)
                    assert np.array_equal(z,q)
                    h=match(s,n,q,g);vec=dict(original=q,guard=g,matched=h)
                    for k,v in vec.items():
                        assert abs(np.linalg.norm(v)/np.linalg.norm(n)-1)<1e-6 and (v@n)/(np.linalg.norm(n)**2)>=.5-1e-7
                        plans[k].append(v)
                    geom.append({k:drift(s,n,v) for k,v in vec.items()})
                    return q
                t.solve=design
                try:y0,_=stream(t,old,edited,tail,True,'batch_observed')
                finally:t.solve=initial
                results={}
                for k,plan in plans.items():
                    ys=[apply(t,c,tail,plan) for c in (old,edited)]
                    if k=='original':
                        assert np.array_equal(ys[1].astype(np.float64),y0)
                        for l,m in ((0,'unwritten'),(1,'normalized')):
                            prev=original[r['id']]['claims'][ci]['arms'][j][m]
                            assert int(ys[l].argmax())==prev['top'] and np.array_equal(ys[l][labels],prev['label_logits'])
                    results[k]=[dict(top=int(v.argmax()),label=int(v[labels].argmax()),logits=v[labels].tolist()) for v in ys]
                entries.append(dict(claim=ci,edition=j,gold=gold,previous=original[r['id']]['claims'][ci]['arms'][j]['old']['class_id'],outputs=results,drift=geom))
        with dest.open('a') as f:f.write(json.dumps(dict(id=r['id'],page=r['page'],entries=entries))+'\n')
        mx.clear_cache()
        if (ix+1)%4==0:print(ix+1,round(time.monotonic()-start,1),flush=True)
        if time.monotonic()-start>1200:raise TimeoutError('guarded reader')
if __name__=='__main__':main()
