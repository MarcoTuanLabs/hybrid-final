"""Score exact wording and prepare a method-blind archive for semantic reading."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
import re
import string

HERE=Path(__file__).resolve().parent


def words(s):
    return re.sub(r'\b(?:a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split()


def f1(a,b):
    if not a or not b:return float(a==b)
    shared=sum((Counter(a)&Counter(b)).values())
    return 2*shared/(len(a)+len(b))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("data",type=Path)
    args=parser.parse_args()
    with args.data.open("rb") as f:users=[json.loads(f.readline()) for _ in range(4)]
    rows=list(map(json.loads,(HERE/"text_rows.jsonl").read_text().splitlines()))
    assert len(rows)==len({tuple(r["key"]) for r in rows})==8
    pool={}; observations={}
    for r in rows:
        u,event,at=r["key"]
        qs=users[u]["sessions"][at]["questions"]
        for mode,arm in r["arms"].items():
            for pos,(j,reply) in enumerate(zip(r["questions"],arm["replies"],strict=True)):
                q=qs[j];gold=q["answer"]
                record={"question":q["question"],"reference":gold,"response":reply}
                k=hashlib.sha256(("hmem-g1"+json.dumps(record,sort_keys=True)).encode()).hexdigest()[:12]
                assert k not in pool or pool[k]==record
                pool[k]=record
                observations[u,event,at,pos,mode]=(k,words(reply)==words(gold),f1(words(reply),words(gold)))
    bank=sorted((dict(id=k,**v) for k,v in pool.items()),key=lambda z:z["id"])
    (HERE/"blind_responses.json").write_text(json.dumps(bank,indent=2)+"\n")
    grades=HERE/"blind_grades.json"
    decisions=json.loads(grades.read_text()) if grades.exists() else {}
    if decisions:assert set(decisions)==set(pool) and all(x in (0,1) for x in decisions.values())
    summary={}
    for mode in ("old","current","notice"):
        exact=[];token_f1=[];semantic=[]
        for j in range(2):
            entries=[v for (*_,pos,m),v in observations.items() if pos==j and mode==m]
            assert len(entries)==8
            exact.append(sum(v[1] for v in entries));token_f1.append(mean(v[2] for v in entries))
            if decisions:semantic.append(sum(decisions[v[0]] for v in entries))
        summary[mode]=dict(exact=exact,token_f1=token_f1,semantic_correct=semantic,
                           mean_complete_s=mean(r["arms"][mode]["complete_s"] for r in rows))
    if decisions:
        joint={}
        for mode in summary:
            joint[mode]=sum(all(decisions[observations[*r["key"],j,mode][0]] for j in range(2)) for r in rows)
        for m,n in joint.items():summary[m]["both_correct"]=n
    out=dict(users=4,episodes=8,questions=16,distinct_responses=len(pool),methods=summary,
             semantic_evaluator="Single method-blind Codex AI review against supplied references" if decisions else None)
    (HERE/"text_summary.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out))


if __name__=="__main__":
    main()
