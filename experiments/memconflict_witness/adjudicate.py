"""Prepare de-identified predictions, then pool frozen semantic grades by persona."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/(name+'.json')).read_text())
def save(name,obj):(HERE/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def compact(text):return ' '.join(text.casefold().split())

def prepare():
    rows=list(map(json.loads,(HERE/'answers.jsonl').read_text().splitlines()))
    qs=read('questions');labels=read('labels')
    assert len(rows)==len({v['id'] for v in rows})==len(qs)==24
    extra={v['id']:v['answers'] for v in map(json.loads,(HERE/'boundary_answers.jsonl').read_text().splitlines())}
    last={v['id']:v['answers'] for v in map(json.loads,(HERE/'latest_answers.jsonl').read_text().splitlines())}
    assert set(extra)==set(last)=={v['id'] for v in qs}
    for uid,answers in last.items():extra[uid].update(answers)
    by_uid={v['id']:v for v in qs};blind={};links={}
    for row in rows:
        uid=row['id'];q=by_uid[uid];key=hashlib.sha256(('mc-blind-1/'+uid).encode()).hexdigest()[:8]
        vals={};links[uid]={}
        for name,text in (row['answers']|extra[uid]).items():
            a=hashlib.sha256((uid+'|'+compact(text)).encode()).hexdigest()[:8]
            vals[a]=text;links[uid][name]=a
        blind[key]=dict(question=q['question'],reference=labels[uid]['answer'],rubric=labels[uid]['category'],
                        answers=dict(sorted(vals.items())))
    save('blind',dict(sorted(blind.items())));save('linkage',links)

def score():
    q=read('questions');labels=read('labels');link=read('linkage');grades=read('grades');blind=read('blind')
    allids={a for d in blind.values() for a in d['answers']}
    assert set(grades)==allids
    for d in blind.values():
        for a in d['answers']:
            allowed=[0,1] if d['rubric']=='conditional_conflict' else [0,.5,1]
            assert grades[a] in allowed
    people=sorted({v['persona'] for v in q});uids=[v['id'] for v in q];methods=sorted(link[uids[0]])
    per={m:{i:grades[link[i][m]] for i in uids} for m in methods}
    stats={}
    for m,d in per.items():
        by_type={}
        for typ in {v['category'] for v in labels.values()}:
            values=[v for i,v in d.items() if labels[i]['category']==typ]
            assert len(values)==8
            by_type[typ]=dict(full=values.count(1),mean=statistics.mean(values))
        stats[m]=dict(full=sum(v==1 for v in d.values()),mean=statistics.mean(d.values()),by_type=by_type)
    baselines=['standard','diverse','wide','wide_typed','witness_typed','ordinary_critique','witness','boundaries','boundaries_plus_witness','latest_session']
    tests={'witness':['standard','diverse','wide','wide_typed','witness_typed','ordinary_critique','boundaries','boundaries_plus_witness','latest_session'],
           'whole':baselines,'recurrent':baselines+['whole']}
    gates={m:stats[m]['full']>=3+max(stats[x]['full'] for x in others) and stats[m]['mean']>=max(stats[x]['mean'] for x in others) for m,others in tests.items()}
    rng=random.Random(7410);draws=[rng.choices(people,k=len(people)) for _ in range(4000)]
    paired={}
    for m,other in [('witness','diverse'),('whole','witness_typed'),('recurrent','whole'),('recurrent','ordinary_critique')]:
        unit={p:sum(int(per[m][i]==1)-int(per[other][i]==1) for i in uids if labels[i]['persona']==p)/3 for p in people}
        delta=sorted(statistics.mean(unit[p] for p in sample) for sample in draws)
        paired[m+' minus '+other]=dict(fraction=statistics.mean(unit.values()),persona_bootstrap95=[delta[99],delta[3899]],
            full_gains=sum(per[m][i]==1 and per[other][i]!=1 for i in uids),
            full_losses=sum(per[m][i]!=1 and per[other][i]==1 for i in uids),
            grade_changes=sum(per[m][i]!=per[other][i] for i in uids),
            different_answers=sum(link[i][m]!=link[i][other] for i in uids))
    rows=list(map(json.loads,(HERE/'answers.jsonl').read_text().splitlines()))
    boundary=list(map(json.loads,(HERE/'boundary_answers.jsonl').read_text().splitlines()))
    last=list(map(json.loads,(HERE/'latest_answers.jsonl').read_text().splitlines()))
    result=dict(latest_positions=sum(v['positions'] for v in last),latest_seconds=sum(v['seconds'] for v in last),boundary_positions=sum(v['positions'] for v in boundary),boundary_seconds=sum(v['seconds'] for v in boundary),n=len(q),personas=len(people),distinct_answers=len(grades),scores=stats,paired=paired,gate=gates,
                positions=sum(v['positions'] for v in rows),seconds=sum(v['seconds'] for v in rows),
                generator='Qwen3.5-4B BF16',judge='One policy-blind AI editorial assessment using the official category-specific answer criteria.')
    save('summary',result);print(json.dumps(result))
if __name__=='__main__':
    ps=argparse.ArgumentParser(description=__doc__);ps.add_argument('action',choices=['prepare','score'])
    if ps.parse_args().action=='prepare':prepare()
    else:score()
