"""Measure the same frozen text-only guard against full rereading without saved KV state."""
import argparse
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
from guard import guarded

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_refresh'))
from run import Engine,encode,decode,make_prompt_cache

def execute(engine,row,mode):
    tick=time.monotonic();pos=engine.positions
    ids,_=encode(engine,row,'old')
    old,_,_=decode(engine,ids,make_prompt_cache(engine.model),0)
    old_positions=engine.positions-pos;old_s=time.monotonic()-tick
    new=None;reason='replay'
    if mode=='guarded_unique':
        new,reason=guarded(row['old'],row['current'],old,'unique',[])
    if new is None:
        ids,_=encode(engine,row,'current')
        new,_,_=decode(engine,ids,make_prompt_cache(engine.model),0)
    mx.synchronize()
    return dict(old=old.strip(),current=new.strip(),old_positions=old_positions,total_positions=engine.positions-pos,
                old_s=old_s,seconds=time.monotonic()-tick,decision=reason,checkpoints=0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path)
    p.add_argument('inputs',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    rows=json.loads(a.inputs.read_text());modes=['full','guarded_unique']
    seen={r['id'] for r in map(json.loads,a.out.read_text().splitlines())} if a.out.exists() else set()
    start=time.monotonic()
    with a.out.open('a') as f:
        for j,row in enumerate(rows):
            if row['id'] in seen:continue
            pol={}
            for mode in modes[::1 if j%2==0 else -1]:
                mx.clear_cache();pol[mode]=execute(engine,row,mode)
            f.write(json.dumps(dict(id=row['id'],policies=pol),ensure_ascii=False)+'\n');f.flush()
            if time.monotonic()-start>1800:raise TimeoutError('registered thirty-minute budget')
if __name__=='__main__':main()
