"""Score a complete set of timed, paired first-token services."""
import argparse
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent

def summarize(contract,out,part):
    c=json.loads(contract.read_text());seeds=c[part+'_seeds']
    rows=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()]
    required={(s,surf,n,a) for s in seeds for surf in ('table','prose') for n in (False,True) for a in (False,True)}
    assert {tuple(r['key']) for r in rows}==required and len(rows)==len(required)
    modes=c[part+'_arms'];primary=c['primary'];refs=c.get('references',['joint','window'])
    assert all(sorted(r['arms'])==sorted(r['order'])==sorted(modes) for r in rows)
    target=lambda r:r['key'][3] and not r['key'][2]
    correct=lambda r,m:r['arms'][m]['top']==r['donor' if target(r) else 'recipient']
    rng=random.Random(73261)
    draws=[rng.choices(seeds,k=len(seeds)) for _ in range(5000)]
    times={m:{s:sum(r['arms'][m]['seconds'] for r in rows if r['key'][0]==s) for s in seeds} for m in modes}
    def total(ss,m):return sum(times[m][i] for i in ss)
    def paired(m,b):
        ratios=sorted(total(d,m)/total(d,b) for d in draws)
        return dict(ratio=total(seeds,m)/total(seeds,b),ci95=[ratios[124],ratios[4874]])
    result={}
    for m in modes:
        result[m]=dict(target=sum(target(r) and correct(r,m) for r in rows),controls=sum(not target(r) and correct(r,m) for r in rows),
                       loss_of_any_ref=sum(target(r) and any(correct(r,b) for b in refs) and not correct(r,m) for r in rows),
                       mean_seconds=total(seeds,m)/len(rows),sum_seconds=total(seeds,m),
                       target_mean_seconds=sum(r['arms'][m]['seconds'] for r in rows if target(r))/(2*len(seeds)),
                       all_top_agree_joint=sum(r['arms'][m]['top']==r['arms']['joint']['top'] for r in rows),
                       writes_equal_joint=sum(r['arms'][m]['write_digest']==r['arms']['joint']['write_digest'] for r in rows),
                       per_reference_time={b:paired(m,b) for b in refs})
    q=result[primary];gate=q['per_reference_time']['joint']['ratio']<=.8 and q['per_reference_time']['window']['ratio']<1 and q['loss_of_any_ref']==0 and q['controls']==len(seeds)*6
    return dict(part=part,seeds=len(seeds),requests=len(rows),targets=len(seeds)*2,controls=len(seeds)*6,methods=result,meets_gate=gate)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('contract',type=Path);p.add_argument('out',type=Path);p.add_argument('part',choices=['pilot','confirm']);a=p.parse_args()
    res=summarize(a.contract,a.out,a.part);(a.out/'summary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
