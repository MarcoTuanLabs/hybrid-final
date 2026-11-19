"""Identify independently labeled paired claims without multi-claim or edit-size filtering."""
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vitaminc_read'))
from data import digest,LABELS
PINS={"dev":"544934677f5d133873e6d38f4557f8966f4efa5d3d70874ffe6913f2091b86b5","train":"7461c6fd1a13459590317c5ccdc8651dd2daf7c1ad8ae4b10ccd88d164fccd5a"}


def corpus(path):
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==PINS[path.stem]
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
    for line in path.open():
        r=json.loads(line)
        if r['revision_type']=='real':
            groups[r['page'],str(r['wiki_revision_id']),r['case_id']][r['evidence']][r['claim']].add(r['label'])
    pooled=defaultdict(set)
    for (pg,rev,_),tx in groups.items():
        if len(tx)!=2:continue
        a,b=sorted(tx)
        for c in tx[a].keys() & tx[b].keys():
            if len(tx[a][c])==len(tx[b][c])==1:
                pooled[pg,rev,a,b,c].add((LABELS.index(next(iter(tx[a][c]))),LABELS.index(next(iter(tx[b][c])))))
    rows=[]
    for (pg,rev,a,b,c),labs in pooled.items():
        if len(labs)!=1:continue
        y=next(iter(labs));rows.append(dict(id=digest('single',pg,rev,a,b,c),page=digest('page',pg),passages=[a,b],claim=c,y=list(y)))
    return rows


def known():
    sp=json.loads((HERE.parent/'vitaminc_read/split.json').read_text())
    return {r['page'] for part in sp.values() for r in part}


if __name__=='__main__':
    for f in sys.argv[1:]:
        rows=corpus(Path(f))
        blocked=known()
        rows=[r for r in rows if r['page'] not in blocked];by=defaultdict(set)
        for r in rows:by[tuple(sorted(r['y']))].add(r['page'])
        print(Path(f).stem,len(rows),{str(k):len(v) for k,v in by.items()})


def draw(rows,counts,excluded):
    by=defaultdict(list)
    for r in rows:
        if r['page'] not in excluded:by[tuple(sorted(r['y']))].append(r)
    used=set(excluded);out=[]
    for key,amount in counts:
        candidates=sorted(by[key],key=lambda r:digest('stratum-v2',r['id']))
        taken=0
        for r in candidates:
            if r['page'] in used:continue
            out.append(r);used.add(r['page']);taken+=1
            if taken==amount:break
        assert taken==amount,(key,taken,amount)
    return sorted(out,key=lambda r:r['id'])


def build(directory):
    pilot=draw(corpus(directory/'dev.jsonl'),[((2,2),1),((0,0),7),((1,2),4),((0,2),16),((0,1),16)],known())
    blocked=known()|{r['page'] for r in pilot}
    fit=draw(corpus(directory/'train.jsonl'),[((2,2),8),((1,1),8),((0,0),24),((1,2),16),((0,2),92),((0,1),92)],blocked)
    return {"fit":fit,"pilot":pilot}


def public_split(parts):
    return {k:[dict(id=r['id'],page=r['page']) for r in rows] for k,rows in parts.items()}
