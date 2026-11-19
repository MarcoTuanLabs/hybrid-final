"""Score paired target changes, preservation, write parity and complete service time."""
import argparse
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('part',choices=['pilot','confirm']);p.add_argument('out',type=Path);a=p.parse_args()
    contract=json.loads((HERE/'contract.json').read_text());seeds=contract[a.part+'_seeds']
    rows=[json.loads(l) for l in (a.out/'rows.jsonl').read_text().splitlines()]
    keys={tuple(r['key']) for r in rows}
    assert keys=={(s,v,c,w) for s in seeds for v in ('table','prose') for c in (True,False) for w in (True,False)} and len(keys)==len(rows)
    modes=sorted(rows[0]['arms']);assert all(sorted(r['arms'])==modes and sorted(r['order'])==modes for r in rows)
    gold=lambda r:r['donor'] if r['key'][3] and not r['key'][2] else r['recipient']
    correct=lambda r,m:r['arms'][m]['top']==gold(r)
    target=[r for r in rows if r['key'][3] and not r['key'][2]]
    controls=[r for r in rows if r not in target]
    times={m:{s:sum(r['arms'][m]['seconds'] for r in rows if r['key'][0]==s) for s in seeds} for m in modes}
    rng=random.Random(88213000)
    draws=[rng.choices(seeds,k=len(seeds)) for _ in range(5000)]
    def total(xs,m):return sum(times[m][s] for s in xs)
    def ratio(m,b):
        boot=sorted(total(s,m)/total(s,b) for s in draws)
        return dict(ratio=total(seeds,m)/total(seeds,b),ci95=[boot[124],boot[4874]])
    summary={}
    for m in modes:
        same=lambda field:sum(r['arms'][m][field]==r['arms']['joint'][field] for r in rows)
        subgroup={s:sum(r['arms'][m]['seconds'] for r in target if r['key'][0]==s) for s in seeds}
        summary[m]=dict(target=sum(correct(r,m) for r in target),controls=sum(correct(r,m) for r in controls),
            retained_all_reference_successes=all(correct(r,m) for r in target if any(correct(r,b) for b in ('cut','joint'))),
            wins_vs_joint=sum(correct(r,m) and not correct(r,'joint') for r in target),losses_vs_joint=sum(not correct(r,m) and correct(r,'joint') for r in target),
            all_top_agree_joint=same('top'),writes_equal_joint=same('write_digest'),original_equal_joint=same('original_digest'),queries_equal_joint=same('query_digest'),
            seconds=total(seeds,m),mean_seconds=total(seeds,m)/len(rows),written_target_mean_seconds=sum(subgroup.values())/len(target),
            time_vs_joint=ratio(m,'joint'),time_vs_cut=ratio(m,'cut'),max_logit_diff=max(r['arms'][m]['max_logit_vs_joint'] for r in rows))
    score=summary['window_last']
    ok=max(score['time_vs_joint']['ratio'],score['time_vs_cut']['ratio'])<=0.80 and score['retained_all_reference_successes'] and score['controls']==len(controls)
    out=dict(part=a.part,seeds=len(seeds),requests=len(rows),written_targets=len(target),preservation=len(controls),methods=summary,meets_gate=ok)
    (a.out/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
