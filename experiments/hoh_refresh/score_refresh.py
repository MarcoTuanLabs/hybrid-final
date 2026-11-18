"""Summarize independently executed paired lifecycles and the registered promotion gate."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_updates'))
from score import evaluate

def main():
    labels=json.loads((HERE.parent/'hoh_updates'/'labels.json').read_text())
    ids=[r['id'] for r in json.loads((HERE.parent/'hoh_updates'/'questions.json').read_text())]
    rows=list(map(json.loads,(HERE/'rows.jsonl').read_text().splitlines()))
    assert [r['id'] for r in rows]==ids
    metrics={}
    for mode in ['full','prefix','text','unique','native','stale']:
        ems=[];f1s=[];first=[];same=[];positions=0;seconds=0.;old_seconds=0.;decisions=Counter()
        for r in rows:
            uid=r['id'];p=r['policies']['full' if mode=='stale' else mode]
            stale=mode=='stale'
            answer=p['old' if stale else 'current']
            em,f1=evaluate(answer,labels[uid]['current'])
            ems.append(em);f1s.append(f1);first.append(evaluate(p['old'],labels[uid]['previous'])[0])
            same.append(answer==r['policies']['full']['current'])
            positions+=p['old_positions' if stale else 'total_positions']
            seconds+=p['old_s' if stale else 'seconds'];old_seconds+=p['old_s']
            decisions['reuse' if stale else p['decision']]+=1
        metrics[mode]=dict(em=sum(ems),f1=statistics.mean(f1s),old_em=sum(first),same_as_full=sum(same),
                           positions=positions,seconds=seconds,old_seconds=old_seconds,decisions=dict(decisions))
    base=metrics['prefix']
    def quality(m,b):
        return m['em']>=b['em'] and m['f1']>=b['f1']
    efficient={}
    for k in ('text','unique','native'):
        m=metrics[k]
        m['position_reduction_vs_prefix']=1-m['positions']/base['positions']
        m['time_reduction_vs_prefix']=1-m['seconds']/base['seconds']
        efficient[k]=m['position_reduction_vs_prefix']>=.20 and quality(m,base)
    candidate=metrics['native'];text=[metrics[k] for k in ('text','unique')]
    acceptable_text=[m for m in text if quality(m,base)]
    extra=(candidate['em']>=max(m['em'] for m in text)+2 or
           (acceptable_text and all(quality(candidate,m) for m in text) and
            candidate['seconds']<=.9*min(m['seconds'] for m in acceptable_text)))
    gate=bool(efficient['native'] and extra)
    summary=dict(n=len(ids),policy=metrics,efficient=efficient,incremental_native_gate=gate,
                 measured_lifecycles=len(rows)*5,
                 native_exclusive_repairs=sum(r['policies']['native']['decision']=='native' for r in rows),
                 first_answer_disagreements=sum(len({m['old'] for m in r['policies'].values()})>1 for r in rows))
    reader=list(map(json.loads,(HERE/'reader_rows.jsonl').read_text().splitlines()))
    assert [row['id'] for row in reader]==ids
    summary['cpu_full']=dict(current_em=sum(evaluate(row['answers']['current'],labels[row['id']]['current'])[0] for row in reader),
                             old_em=sum(evaluate(row['answers']['old'],labels[row['id']]['previous'])[0] for row in reader),
                             f1=statistics.mean(evaluate(row['answers']['current'],labels[row['id']]['current'])[1] for row in reader),
                             wall_s=sum(row['seconds'] for row in reader))
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
