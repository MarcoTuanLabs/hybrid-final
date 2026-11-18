"""Compare fixed, query-independent update selections at matched whole-turn budgets."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
read=lambda f:json.loads((HERE/f).read_text())


def select(order,turns,budget):
    included=[];spent=0
    for j in order:
        n=turns[j]["tokens"]
        if spent+n<=budget:included.append(j);spent+=n
    return sorted(included),spent


def uniform(n):
    # Traverse a balanced bisection, then keep the highest-priority turns that fit.
    order=[];queue=[(0,n)]
    while queue:
        low,high=queue.pop(0)
        if high<=low:continue
        mid=(low+high)//2;order.append(mid)
        queue.extend(((low,mid),(mid+1,high)))
    return order


def main():
    cases=read("raw_inputs.json");shallow=read("shallow_scores.json")["sessions"];dense=read("dense_scores.json")
    assert set(cases)==set(shallow)==set(dense)
    # Selection is complete before the evaluation-only references and witnesses are loaded.
    methods=["latest","uniform","user_recent","idf","dense","raw","normalized"];sel={}
    for k,c in cases.items():
        turns=c["turns"];n=len(turns);ids=list(range(n))
        vocab=[set(re.findall(r"[a-z0-9]+",t["text"].lower())) for t in turns]
        df=Counter(w for v in vocab for w in v)
        def priority(vals):
            return sorted(ids,key=lambda i:(-vals[i]/math.sqrt(turns[i]["tokens"]),i))
        idf=[sum(math.log((n+1)/(df[w]+1))+1 for w in vs)/max(len(vs),1) for vs in vocab]
        scores=shallow[k]["scores"]
        orders=dict(latest=ids[::-1],uniform=uniform(n),user_recent=sorted(ids,key=lambda i:(turns[i]["role"]!="user",-i)),
                    idf=priority(idf),dense=priority(dense[k]["scores"]),raw=priority([s[1] for s in scores]),
                    normalized=priority([s[0] for s in scores]))
        assert all(sorted(order)==ids for order in orders.values())
        sel[k]=dict(budget=c["budget"],total=sum(t["tokens"] for t in turns),methods={})
        for m in methods:
            chosen,spent=select(orders[m],turns,c["budget"])
            sel[k]["methods"][m]=dict(indices=chosen,tokens=spent)
    (HERE/"selections.json").write_text(json.dumps(sel,indent=2)+"\n")
    targets=read("witnesses.json");eligible=[v for v in targets.values() if v["witnesses"]]
    scores={};covered={};users={v["user"] for v in eligible}
    for m in methods:
        covered[m]={q for q,v in targets.items() if set(v["witnesses"]) & set(sel[v["event"]]["methods"][m]["indices"])}
    for m in methods:
        scores[m]=dict(covered=len(covered[m]),normalized_wins=len(covered["normalized"]-covered[m]),
                       normalized_losses=len(covered[m]-covered["normalized"]),
                       selected_tokens=sum(x["methods"][m]["tokens"] for x in sel.values()))
    best=max(scores[m]["covered"] for m in ("latest","uniform","user_recent","idf","dense"));s=scores["normalized"]["covered"];total=len(eligible)
    result=dict(sessions=len(cases),questions=len(targets),attainable=total,eligible_users=len(users),original_tokens=sum(x["total"] for x in sel.values()),
                methods=scores,promote=total>=8 and len(users)>=3 and s>=.8*total and s>=best+2 and s>=scores["raw"]["covered"]+1)
    (HERE/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result))


if __name__=="__main__":main()
