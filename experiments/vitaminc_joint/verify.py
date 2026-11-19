"""Independently recompute page-level totals from public raw predictions using standard Python."""
import hashlib
import json
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
EARLIER=HERE.parent/'vitaminc_read'
def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]

def flat(x):
    if isinstance(x,int):return [x]
    return [y for item in x for y in flat(item)]

def check(sp,answers,summary):
    assert [(r['id'],r['page']) for r in answers]==[(r['id'],r['page']) for r in sp]
    for k,s in summary['methods'].items():
        n=changed=stable=both=0
        for r in answers:
            g=flat(r['gold']);v=flat(r['outputs'][k]);assert len(g)==len(v) and len(g)%2==0
            assert all(x in (0,1,2) for x in g+v)
            for i in range(0,len(g),2):
                ns=sum(int(x==y) for x,y in zip(g[i:i+2],v[i:i+2]))
                n+=ns;both+=int(ns==2)
                if g[i]==g[i+1]:stable+=ns
                else:changed+=ns
        assert (n,changed,stable,both)==(s['correct'],s['changed'],s['stable'],s.get('both',s.get('pair_both'))),(k,s)

def audit():
    sets=[]
    for d in (EARLIER,HERE):
        for f,h in read(d/'frozen_files.json').items():
            assert hashlib.sha256((d/f).read_bytes()).hexdigest()==h
        sp=read(d/'split.json')
        for rr in sp.values():
            pages=[r['page'] for r in rr];assert len(pages)==len(set(pages))
            sets.append(set(pages))
    assert sum(map(len,sets))==len(set.union(*sets))==468
    old=read(EARLIER/'split.json')['pilot'];p=EARLIER/'pilot'
    first=rows(p/'rows.jsonl')
    s0=read(p/'summary.json');check(old,first,s0)
    second=rows(p/'component_rows.jsonl')
    for i,r in enumerate(second):
        for k in ('full','paired_labels','old_transition','absolute_labels','full_contrast'):
            r['outputs'][k]=first[i]['outputs'][k]
    for model in ('albert-base-vitaminc-mnli','albert-xlarge-vitaminc'):
        for r,official in zip(second,rows(p/(model+'-pilot.jsonl')),strict=True):
            assert r['id']==official['id'] and r['gold']==official['gold']
            r['outputs'][model]=official['answers']
    s1=read(p/'component_summary.json');check(old,second,s1)
    assert s0['promote'] is False and s1['promote'] is False
    sp=read(HERE/'split.json');r=rows(HERE/'pilot/rows.jsonl');s=read(HERE/'pilot/summary.json')
    check(sp['pilot'],r,s)
    assert s['endpoints']==len(r)*2==88 and s['stable']==sum(z['gold'][0]==z['gold'][1] for z in r)*2==16
    methods=s['methods'];best=max(v['correct'] for k,v in methods.items() if k!='structured')
    assert s['promote']==(methods['structured']['correct']>=best+4 and methods['structured']['stable']>=methods['direct']['stable']-1)
    print('468 distinct page hashes across all splits. All archived correct, changed, stable, and both-edition totals agree.')
if __name__=='__main__':audit()
