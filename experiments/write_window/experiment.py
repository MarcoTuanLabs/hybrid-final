"""Time all candidate and reference services independently in randomized order."""
import argparse
import json
from pathlib import Path
import random
import time
import mlx.core as mx
import numpy as np
from service import validate,execute

HERE=Path(__file__).resolve().parent
MODES=("cut","joint","window","window_last")

def main():
    p=argparse.ArgumentParser();p.add_argument("model",type=Path);p.add_argument("repo",type=Path)
    p.add_argument("part",choices=("smoke","pilot","confirm"));p.add_argument("out",type=Path);a=p.parse_args()
    trial,deps=validate(a.model,a.repo);make=deps[1];mx.set_cache_limit(96*1024**2)
    proto=json.loads((HERE/'contract.json').read_text())
    seeds=proto['pilot_seeds'] if a.part!='confirm' else proto['confirm_seeds']
    modes=MODES if a.part!='confirm' else ("cut","joint","window_last")
    example=make(27112000,"table",False)
    for m in modes:
        warm=execute(trial,example,m,True,deps)
        if a.part=='smoke':print(m,warm[0],flush=True)
    if a.part=='smoke':return
    a.out.mkdir(exist_ok=True,parents=True);dest=a.out/'rows.jsonl'
    seen={tuple(json.loads(r)['key']) for r in dest.read_text().splitlines()} if dest.exists() else set()
    start=time.monotonic()
    with dest.open('a') as f:
        for seed in seeds:
            for surf in ("table","prose"):
                for nuisance in (False,True):
                    ex=make(seed,surf,nuisance)
                    gold=[trial.tok.encode(ex[n],add_special_tokens=False)[0] for n in ('value_a','value_b')]
                    for active in (True,False):
                        key=(seed,surf,nuisance,active)
                        if key in seen:continue
                        order=list(modes)
                        random.Random(seed*41+19*nuisance+37*active+53*(surf=='table')).shuffle(order)
                        results={m:execute(trial,ex,m,active,deps) for m in order}
                        arms={}
                        for m,(rec,y,slot) in results.items():
                            arms[m]=dict(rec,margin=float(y[gold[1]]-y[gold[0]]),max_logit_vs_joint=float(np.max(np.abs(y-results['joint'][1]))),
                                         write_max_abs_vs_joint=float(np.max(abs(slot-results['joint'][2]))))
                        f.write(json.dumps(dict(key=key,order=order,recipient=gold[0],donor=gold[1],arms=arms))+'\n');f.flush()
            print(a.part,seed,round(time.monotonic()-start,1),flush=True);mx.clear_cache()
            if time.monotonic()-start>proto['deadline_seconds'][a.part]:raise TimeoutError(a.part)
if __name__=='__main__':main()
