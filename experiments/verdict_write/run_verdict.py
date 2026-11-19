"""Test the original 256-coordinate write on real, three-way factual judgments."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
from mlx_lm.models.cache import KVCache
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'write_window'))
from service import validate

def main():
    p=argparse.ArgumentParser()
    for n in ('model','repo','data','out'):p.add_argument(n,type=Path)
    a=p.parse_args();repo=a.repo.resolve();t,deps=validate(a.model,repo);_,_,host,set_write,_,stream=deps
    src=repo/'experiments/vitaminc_read'
    sys.path.append(str(src))
    from data import split,manifest
    spec=importlib.util.spec_from_file_location('verification_prompt',src/'run.py')
    prompt=importlib.util.module_from_spec(spec);spec.loader.exec_module(prompt)
    parts=split(a.data);assert manifest(parts)==json.loads((src/'split.json').read_text())
    labels=[t.tok.encode(c,add_special_tokens=False)[0] for c in 'ABC']
    assert all(t.tok.encode(c,add_special_tokens=False)==[v] for c,v in zip('ABC',labels))
    think=t.tok.encode('<think>',add_special_tokens=False)[0]
    def direct(cache,tail):
        v=t.model(mx.array([tail]),cache=copy.deepcopy(cache))[0,-1];return host(v)
    def describe(y):
        assert np.isfinite(y).all()
        return dict(top=int(y.argmax()),class_id=int(np.argmax(y[labels])),label_logits=y[labels].tolist())
    a.out.mkdir(exist_ok=True,parents=True);dest=a.out/'rows.jsonl'
    seen={json.loads(l)['id'] for l in dest.read_text().splitlines()} if dest.exists() else set()
    mx.set_cache_limit(96*1024**2);start=time.monotonic()
    for i,r in enumerate(parts['pilot']):
        if r['id'] in seen:continue
        all_claims=[]
        for claim,gold in zip(r['claims'],r['gold']):
            halves=[];tails=[]
            for passage in r['passages']:
                ids=prompt.tokens(t.tok,passage,claim)
                assert ids.count(think)==1
                n=ids.index(think);assert len(ids)-n==4
                halves.append(t.prefill(ids[:n]));tails.append(ids[n:])
            assert tails[0]==tails[1];tail=tails[0]
            native=[describe(direct(c,tail)) for c in halves]
            results=[]
            for j in (0,1):
                old=halves[1-j];new=halves[j]
                slot=new[28][0][:,-1:,t.lo:t.lo+256];edit=set_write(t,old,slot)
                arms=dict(old=native[1-j],full=native[j])
                arms['write_native']=describe(direct(edit,tail))
                for name,active,reader in [('white',True,'batch_white'),('normalized',True,'batch_observed'),('unwritten',False,'batch_observed')]:
                    y,qs=stream(t,old,edit,tail,active,reader)
                    assert len(qs)==4;arms[name]=describe(y)
                for name in ('recurrent','attention'):
                    mixed=[(b if (isinstance(a0,KVCache)==(name=='attention')) else a0) for a0,b in zip(old,new)]
                    arms[name]=describe(direct(mixed,tail))
                results.append(arms)
            all_claims.append(dict(gold=gold,arms=results))
        with dest.open('a') as f:f.write(json.dumps(dict(id=r['id'],page=r['page'],labels=labels,claims=all_claims))+'\n')
        mx.clear_cache()
        if (i+1)%4==0:print(i+1,round(time.monotonic()-start,1),flush=True)
        if time.monotonic()-start>1000:raise TimeoutError('public transfer')
if __name__=='__main__':main()
