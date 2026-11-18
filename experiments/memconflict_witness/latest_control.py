"""Answer from the last completed encounter alone, an available recency baseline."""
import argparse
import json
import importlib.util
from pathlib import Path
import sys
import time
import mlx.core as mx
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from query import Engine,generate
spec=importlib.util.spec_from_file_location('memconflict_reader',HERE/'run.py')
reader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
TYPED,prompt=reader.TYPED,reader.prompt

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',type=Path);model=parser.parse_args().model
    load=lambda n:json.loads((HERE/(n+'.json')).read_text())
    qs=load('questions');hs=load('histories')
    mx.set_cache_limit(64*1024**2);engine=Engine(model);start=time.monotonic()
    path=HERE/'latest_answers.jsonl'
    done={r['id'] for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    with path.open('a') as f:
        for q in qs:
            uid=q['id']
            if uid in done:continue
            n=engine.positions;t=time.monotonic()
            answer=generate(engine,prompt(q,[hs[uid][-1]],TYPED),80)
            f.write(json.dumps(dict(id=uid,answers={'latest_session':answer},positions=engine.positions-n,seconds=time.monotonic()-t),ensure_ascii=False)+'\n')
            f.flush();mx.clear_cache()
            if time.monotonic()-start>900:raise TimeoutError('Latest-session comparator')
if __name__=='__main__':main()
