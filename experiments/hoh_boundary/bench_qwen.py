"""Measure guarded answer repair and matched hybrid-model baselines independently."""
import argparse
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
from guard import guarded,revise

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_refresh'))
import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path);p.add_argument('inputs',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--native',action='store_true')
    a=p.parse_args();items=json.loads(a.inputs.read_text())
    mx.set_cache_limit(64*1024**2);engine=run.Engine(a.model)
    modes=['prefix','unique','guarded_unique']
    if a.native:modes.append('guarded_native')
    seen={r['id'] for r in map(json.loads,a.out.read_text().splitlines())} if a.out.exists() else set()
    start=time.monotonic()
    with a.out.open('a') as f:
        for j,row in enumerate(items):
            if row['id'] in seen:continue
            policies={}
            for name in modes[j%len(modes):]+modes[:j%len(modes)]:
                mx.clear_cache();run.revise=guarded if name.startswith('guarded_') else revise
                policies[name]=run.execute(engine,row,name.removeprefix('guarded_'))
            f.write(json.dumps(dict(id=row['id'],policies=policies),ensure_ascii=False)+'\n');f.flush()
            print(j+1,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('registered thirty-minute budget')

if __name__=='__main__':main()
