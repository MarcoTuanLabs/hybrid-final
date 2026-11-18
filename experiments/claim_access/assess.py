"""Blind distinct generated answers for assessment and score paired outcomes by persona."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def load_cases(source):return {q['id']:q for q in read(source/'questions.json')}
def answer_id(uid,text):
    return hashlib.sha256((uid+'|'+' '.join(text.casefold().split())).encode()).hexdigest()[:8]
def prepare(source):
    qs=load_cases(source);labels=read(source/'labels.json');cases={};link={}
    prev=read(source/'grades.json') if (source/'grades.json').exists() else {}
    old=read(source/'linkage.json') if (source/'linkage.json').exists() else {}
    earlier={}
    if old:
        for slot in ['answers','boundary_answers','latest_answers']:
            for row in map(json.loads,(source/(slot+'.jsonl')).read_text().splitlines()):
                earlier.setdefault(row['id'],{}).update(row['answers'])
    rows=list(map(json.loads,(HERE/'answers.jsonl').read_text().splitlines()))
    assert len(rows)==len({r['id'] for r in rows})==len(qs)
    for r in rows:
        uid=r['id'];key=hashlib.sha256(('access-blind/'+uid).encode()).hexdigest()[:8]
        all_answers={('prior_'+k):v for k,v in earlier.get(uid,{}).items()}|r['answers']
        links={k:answer_id(uid,v) for k,v in all_answers.items()}
        for k,aid in old.get(uid,{}).items():assert aid==links['prior_'+k]
        link[uid]=links;cases[key]=dict(question=qs[uid]['question'],reference=labels[uid]['answer'],rubric=labels[uid]['category'],answers={links[k]:v for k,v in all_answers.items()})
    write(HERE/'linkage.json',link);write(HERE/'blind.json',cases)
    fresh={k:dict(v,answers={a:t for a,t in v['answers'].items() if a not in prev}) for k,v in cases.items()}
    write(HERE/'blind_new.json',fresh)
    print('Distinct new responses',sum(len(v['answers']) for v in fresh.values()),'across',len(fresh),'questions')
def compute(source,grades):
    qs=load_cases(source);labels=read(source/'labels.json');link=read(HERE/'linkage.json')
    blind=read(HERE/'blind.json')
    ids=set(a for b in blind.values() for a in b['answers']);assert ids<=set(grades)
    for case in blind.values():
        allowed={0,1} if case['rubric']=='conditional_conflict' else {0,.5,1}
        assert all(grades[k] in allowed for k in case['answers'])
    methods=sorted(next(iter(link.values())));people=sorted(set(q['persona'] for q in qs.values()))
    assert all(sorted(v)==methods for v in link.values())
    by={m:{q:grades[link[q][m]] for q in qs} for m in methods};stats={}
    for m,values in by.items():
        grouped={kind:[s for q,s in values.items() if labels[q]['category']==kind] for kind in sorted(set(v['category'] for v in labels.values()))}
        stats[m]=dict(full=sum(v==1 for v in values.values()),mean=statistics.mean(values.values()),by_type={k:v.count(1) for k,v in grouped.items()})
    base=[m for m in methods if m.startswith('prior_')];new_text=[m for m in methods if m not in base+['matrix_gain']]
    strongest=lambda subset:max(subset,key=lambda m:(stats[m]['full'],stats[m]['mean'],m))
    incumbent=strongest(base) if base else None;best=strongest(new_text)
    comp=strongest(new_text+base);ref=incumbent or best
    rng=random.Random(18236);draws=[rng.choices(people,k=len(people)) for _ in range(5000)]
    def paired(m,b):
        diffs={q:int(by[m][q]==1)-int(by[b][q]==1) for q in qs}
        units={p:statistics.mean(v for q,v in diffs.items() if qs[q]['persona']==p) for p in people}
        boost=sorted(statistics.mean(units[p] for p in sample) for sample in draws)
        return dict(difference=statistics.mean(units.values()),ci95=[boost[124],boost[4874]],gains=list(diffs.values()).count(1),losses=list(diffs.values()).count(-1))
    return dict(n=len(qs),personas=len(people),scores=stats,incumbent=incumbent,best_text_or_whole=best,comparisons={m+' minus '+b:paired(m,b) for m,b in [(best,ref),('matrix_gain',comp),('matrix_gain','whole_gain'),('matrix_gain','lm')]},
                gates=dict(conventional=(bool(incumbent) and stats[best]['full']>=stats[ref]['full']+3 and stats[best]['mean']>=stats[ref]['mean']),recurrent=(stats['matrix_gain']['full']>=stats[comp]['full']+3 and stats['matrix_gain']['mean']>=max(stats[m]['mean'] for m in new_text+base))))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['prepare','score']);ap.add_argument('data',type=Path);a=ap.parse_args()
    if a.phase=='prepare':return prepare(a.data)
    frozen=read(a.data/'grades.json') if (a.data/'grades.json').exists() else {}
    fresh=read(HERE/'new_grades.json')
    assert not set(fresh)&set(frozen)
    out=compute(a.data,frozen|fresh);write(HERE/'summary.json',out);print(json.dumps(out))
if __name__=='__main__':main()
