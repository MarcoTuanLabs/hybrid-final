"""Compare online lifecycle costs with paired page bootstrap intervals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_updates'))
from score import evaluate

def interval(values):
    ordered=sorted(values)
    return [ordered[round((len(ordered)-1)*f)] for f in [.025,.975]]

def summarize(path,labels,ids,base):
    rows=list(map(json.loads,path.read_text().splitlines()))
    assert [r['id'] for r in rows]==ids and len(set(ids))==len(ids)==len(labels)
    names=sorted(rows[0]['policies']);metrics={};ind={}
    for name in names:
        a=[r['policies'][name] for r in rows]
        vals=[evaluate(v['current'],labels[r['id']]['current']) for r,v in zip(rows,a,strict=True)]
        em,f1=map(list,zip(*vals,strict=True))
        times=[v['seconds'] for v in a];pos=[v['total_positions'] for v in a]
        ind[name]=(em,f1,times,pos)
        metrics[name]=dict(em=sum(em),f1=statistics.mean(f1),seconds=sum(times),positions=sum(pos),
                           old_em=sum(evaluate(v['old'],labels[r['id']]['previous'])[0] for r,v in zip(rows,a,strict=True)),
                           decisions=dict(Counter(v['decision'] for v in a)))
    b=metrics[base]
    rng=random.Random(20261118)
    boots=[rng.choices(range(len(rows)),k=len(rows)) for _ in range(2000)]
    for name,m in metrics.items():
        if name==base:continue
        cost=1-m['positions']/b['positions'];elapsed=1-m['seconds']/b['seconds']
        m.update(position_saving=cost,time_saving=elapsed,
                 gate=cost>=.2 and elapsed>=.2 and m['em']>=b['em'] and m['f1']>=b['f1'])
        em,_,ts,_=ind[name];be,_,bt,_=ind[base]
        m['paired_current_gains']=sum(x>y for x,y in zip(em,be,strict=True))
        m['paired_current_losses']=sum(x<y for x,y in zip(em,be,strict=True))
        m['ci95_em_difference']=interval([sum(em[i]-be[i] for i in sample)/len(sample) for sample in boots])
        m['ci95_time_saving']=interval([1-sum(ts[i] for i in sample)/sum(bt[i] for i in sample) for sample in boots])
    return metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('tag',choices=['dev','test'])
    p.add_argument('--cpu-only',action='store_true')
    a=p.parse_args()
    labelsdir=HERE if a.tag=='test' else HERE.parent/'hoh_updates'
    labels=json.loads((labelsdir/('test_labels.json' if a.tag=='test' else 'labels.json')).read_text())
    ids=list(labels)
    s={'n':len(ids),'cpu':summarize(HERE/(a.tag+'_cpu.jsonl'),labels,ids,'full')}
    if not a.cpu_only:
        s['qwen']=summarize(HERE/(a.tag+'_qwen.jsonl'),labels,ids,'prefix')
        if 'guarded_native' in s['qwen']:
            g=s['qwen']['guarded_native'];t=s['qwen']['guarded_unique']
            s['native_advantage']=g['gate'] and (g['em']>=t['em']+2 or (
                g['em']>=t['em'] and g['f1']>=t['f1'] and g['seconds']<=.9*t['seconds']))
    (HERE/(a.tag+('_cpu_summary.json' if a.cpu_only else '_summary.json'))).write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(s,indent=2))

if __name__=='__main__':main()
