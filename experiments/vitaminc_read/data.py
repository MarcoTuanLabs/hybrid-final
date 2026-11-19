"""Sample page-disjoint natural revision pairs from the pinned VitaminC release."""
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path

PIN = "544934677f5d133873e6d38f4557f8966f4efa5d3d70874ffe6913f2091b86b5"
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
SIZES = {"fit": (6,58), "pilot": (5,19), "confirmation": (10,86)}


def digest(*items):
    return hashlib.sha256(json.dumps(items,ensure_ascii=False).encode()).hexdigest()


def split(path):
    with Path(path).open("rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest()==PIN
    versions=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
    for line in Path(path).open():
        r=json.loads(line)
        if r["revision_type"]=="real":
            versions[r["page"],str(r["wiki_revision_id"]),r["case_id"]][r["evidence"]][r["claim"]].add(r["label"])
    pairs=defaultdict(lambda:defaultdict(set))
    for (page,rev,_),texts in versions.items():
        if len(texts)!=2:continue
        a,b=sorted(texts)
        for c in texts[a].keys() & texts[b].keys():
            if len(texts[a][c])==len(texts[b][c])==1:
                pairs[page,rev,a,b][c].add((next(iter(texts[a][c])),next(iter(texts[b][c]))))
    grouped=[defaultdict(list),defaultdict(list)]
    for (page,rev,a,b),cl in pairs.items():
        cs=[(c,next(iter(v))) for c,v in cl.items() if len(v)==1]
        changed=sorted((x for x in cs if x[1][0]!=x[1][1]),key=lambda x:digest("q",x[0]))
        stable=sorted((x for x in cs if x[1][0]==x[1][1]),key=lambda x:digest("q",x[0]))
        if changed and stable:
            group=0;chosen=[changed[0],stable[0]]
        else:
            if len(changed)<2:continue
            op=[v for v in SequenceMatcher(None,a.split(),b.split(),autojunk=False).get_opcodes() if v[0]!="equal"]
            if len(op)!=1:continue
            _,i,j,k,l=op[0]
            if not 1<=j-i<=8 or not 1<=l-k<=8:continue
            group=1;chosen=changed[:2]
        grouped[group][page].append(dict(id=digest("rev",page,rev,a,b),page=digest("page",page),passages=[a,b],claims=[z[0] for z in chosen],gold=[[LABELS.index(v) for v in z[1]] for z in chosen]))
    order=[sorted(g,key=lambda p:digest("order-v1",p)) for g in grouped]
    order[1]=[p for p in order[1] if p not in grouped[0]]
    result={}; starts=[0,0]
    for name,counts in SIZES.items():
        rows=[]
        for g,n in enumerate(counts):
            for page in order[g][starts[g]:starts[g]+n]:
                rows.append(min(grouped[g][page],key=lambda x:x["id"]))
            starts[g]+=n
        assert len(rows)==sum(counts)
        result[name]=sorted(rows,key=lambda r:r["id"])
    assert len({r['page'] for rows in result.values() for r in rows})==sum(map(len,result.values()))
    return result


def manifest(parts):
    return {part:[dict(id=x["id"],page=x["page"],claims=[digest("claim",c) for c in x["claims"]]) for x in rows] for part,rows in parts.items()}
