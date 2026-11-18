import json
import sys
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
import argparse
import gzip
parser=argparse.ArgumentParser()
parser.add_argument('--run-dir',type=Path,default=p.parents[1]/'evidence/quantized_9b')
args=parser.parse_args()
cfg=json.loads((p.parents[1]/'evidence/quantized_9b/contract.json').read_text())
f=args.run_dir/'rows.jsonl'
raw=f.read_text() if f.exists() else gzip.decompress(f.with_suffix('.jsonl.gz').read_bytes()).decode()
rows=list(map(json.loads,raw.splitlines()))
modes=cfg['modes'];seeds=cfg['seeds']
assert len(rows)==len(seeds)*2*2*2*len(modes)
by={(r['seed'],r['surface'],r['nuisance'],r['written'],r['mode']):r for r in rows}
assert len(by)==len(rows)
def group(m,nuisance,written):return [by[s,f,nuisance,written,m] for s in seeds for f in ('table','prose')]
def good(r):return r['top']==r['gold']
stats={}
for m in modes:
    groups=[group(m,n,w) for n,w in ((False,True),(True,True),(False,False),(True,False))]
    costs=groups[0]+groups[1]
    stats[m]=dict(target=sum(map(good,groups[0])),nuisance=sum(map(good,groups[1])),
                  unwritten_target=sum(map(good,groups[2])),unwritten_nuisance=sum(map(good,groups[3])),
                  mean_design_s=float(np.mean([r['design_s'] for r in costs])),
                  mean_charged_s=float(np.mean([r['prefix_s']+(r['geometry_s'] if m!='native' else 0)+r['design_s']+r['rollout_s'] for r in costs])))
def paired(m,other):
    a=group(m,False,True);b=group(other,False,True)
    return dict(wins=sum(good(x) and not good(y) for x,y in zip(a,b)),losses=sum(good(y) and not good(x) for x,y in zip(a,b)))
pairs={m:paired('arc6',m) for m in modes if m!='arc6'}
n=2*len(seeds)
forecast=np.array([max(r['forecast_rel']) for r in rows if r['mode']=='native' and r['written']])
c=stats['arc6'];ref=stats['ref80']
valid=ref['target'] >= stats['native']['target']+4
preserve=all(c[k]>=ref[k] for k in ('nuisance','unwritten_target','unwritten_nuisance'))
passed=valid and preserve and c['target']>=ref['target']-1 and c['target']>=stats['ref16']['target']-1 and c['target']>=stats['white']['target']+2
out=dict(seeds=len(seeds),per_condition=n,live_full_vocab_rollouts=len(rows),scores=stats,arc_vs=pairs,reference_behavior_active=valid,
         median_max_per_case_forecast_relative=float(np.median(forecast)),maximum_forecast_relative=float(max(forecast)),
         decision='confirm_on_64_fresh_seeds' if passed else 'do_not_scale_six_step_reader_from_this_transfer')
print(json.dumps(out,indent=2))
