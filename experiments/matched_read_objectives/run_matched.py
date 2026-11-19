"""Change only how the four query vectors are designed at an identical native write."""
import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from objectives import solve

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    for name in ("model","repo","out"):p.add_argument(name,type=Path)
    p.add_argument("phase",choices=("smoke","pilot","confirm"))
    a=p.parse_args();repo=a.repo.resolve();c=json.loads((HERE/"contract.json").read_text())
    sys.path.insert(0,str(repo/"experiments/write_window"));from service import validate
    sys.path.insert(0,str(repo/"experiments/guarded_read"));from experiment import apply
    a.out.mkdir(parents=True,exist_ok=True);dest=a.out/"rows.jsonl"
    launched=a.out/"wall_start.json"
    if not launched.exists(): launched.write_text(json.dumps(dict(phase=a.phase,start=time.time())))
    run=json.loads(launched.read_text());assert run["phase"]==a.phase
    remaining=c["deadlines"][a.phase] - (time.time()-run["start"])
    if remaining<=0: raise TimeoutError(a.phase)
    def stop(signum,frame): raise TimeoutError(a.phase)
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,remaining)
    t,deps=validate(a.model,repo);_,make,_,set_write,_,stream=deps
    seen={tuple(json.loads(l)["key"]) for l in dest.read_text().splitlines()} if dest.exists() else set()
    start=time.monotonic();original=t.solve;mx.set_cache_limit(96*1024**2)
    for seed in c[a.phase+"_seeds"]:
        for form in ("table","prose"):
            for nuisance in (False,True):
                e=make(seed,form,nuisance);aa,suffix=t.encode(e,"a");bb,tail=t.encode(e,"b");assert suffix==tail
                gold=[t.tok.encode(e[f],add_special_tokens=False)[0] for f in ("value_a","value_b")]
                old=t.prefill(aa);donor=t.prefill(bb)
                edit=set_write(t,old,donor[28][0][:,-1:,t.lo:t.lo+256])
                for active in (True,False):
                    key=(seed,form,nuisance,active)
                    if key in seen:continue
                    slots=[]
                    def capture(slot,mode):
                        slots.append(slot[:3]);return original(slot,mode)
                    t.solve=capture
                    try:incumbent,_=stream(t,old,edit,tail,active,"batch_observed")
                    finally:t.solve=original
                    assert len(slots)==4
                    def outcome(y):
                        assert np.isfinite(y).all()
                        return dict(top=int(y.argmax()),margin=float(y[gold[1]]-y[gold[0]]))
                    outputs={"deployed":outcome(incumbent)}
                    methods=list(c["objectives"]);random.Random(str(key)).shuffle(methods)
                    for m in methods:
                        begin=time.perf_counter();queries=[solve(s,d,n,m) for s,d,n in slots]
                        seconds=time.perf_counter()-begin
                        for q,(_,_,n) in zip(queries,slots,strict=True):
                            assert abs(np.linalg.norm(q)/np.linalg.norm(n)-1)<1e-6 and q@n/(n@n)>=.5-1e-7
                        y=apply(t,edit if active else old,tail,queries)
                        outputs[m]=dict(outcome(y),solver_seconds=seconds,max_logit_vs_deployed=float(np.max(abs(y-incumbent))))
                    with dest.open("a") as file:
                        file.write(json.dumps(dict(key=key,recipient=gold[0],donor=gold[1],methods=outputs))+"\n")
            mx.clear_cache()
        print(a.phase,seed,round(time.monotonic()-start,1),flush=True)
        if time.monotonic()-start>c["deadlines"][a.phase]:raise TimeoutError(a.phase)
if __name__=="__main__":main()
