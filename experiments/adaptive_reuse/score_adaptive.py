"""Select an eligible schedule using exposed labels; report every candidate and both baselines."""
import argparse
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('part',choices=['pilot','confirm']);p.add_argument('out',type=Path);a=p.parse_args()
    c=json.loads((HERE/'contract.json').read_text())
    modes=c[a.part+'_arms'];seeds=c[a.part+'_seeds']
    rows=[json.loads(l) for l in (a.out/'rows.jsonl').read_text().splitlines()];keys={tuple(r['key']) for r in rows}
    assert keys=={(i,x,z,w) for i in seeds for x in ('table','prose') for z in (0,1) for w in (0,1)} and len(keys)==len(rows)
    assert all(sorted(r['order'])==sorted(r['arms'])==sorted(modes) for r in rows)
    target=lambda r:r['key'][3] and not r['key'][2]
    correct=lambda r,m:r['arms'][m]['top']==r['donor' if target(r) else 'recipient']
    refs=c['references']
    sums={m:{i:sum(r['arms'][m]['seconds'] for r in rows if r['key'][0]==i) for i in seeds} for m in modes}
    rng=random.Random(435633);draws=[rng.choices(seeds,k=len(seeds)) for _ in range(5000)]
    def ratio(m,b):
        total=lambda mix,k:sum(sums[k][i] for i in mix)
        boot=sorted(total(s,m)/total(s,b) for s in draws)
        return dict(ratio=total(seeds,m)/total(seeds,b),ci95=[boot[124],boot[4874]])
    stats={}
    for m in modes:
        v=dict(target=sum(target(r) and correct(r,m) for r in rows),controls=sum(not target(r) and correct(r,m) for r in rows),
               losses=sum(target(r) and any(correct(r,b) for b in refs) and not correct(r,m) for r in rows),
               mean_seconds=sum(sums[m].values())/len(rows),per_reference_time={b:ratio(m,b) for b in refs},
               writes_equal_joint=sum(r['arms'][m]['write_digest']==r['arms']['joint']['write_digest'] for r in rows))
        v['eligible']=m in c['candidates'] and v['losses']==0 and v['controls']==len(seeds)*6 and v['per_reference_time']['joint']['ratio']<=.8 and v['per_reference_time']['packed']['ratio']<1
        stats[m]=v
    eligible=sorted((z['mean_seconds'],m) for m,z in stats.items() if z['eligible'])
    s=dict(requests=len(rows),targets=2*len(seeds),controls=6*len(seeds),methods=stats,selected=eligible[0][1] if eligible else None)
    (a.out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s))
if __name__=='__main__':main()
