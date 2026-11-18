"""Score delayed-session services and create a blinded, deduplicated factual reading."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from statistics import mean
import string

HERE=Path(__file__).resolve().parent


def words(text):
    return re.sub(r'\b(?:a|an|the)\b',' ',text.lower().translate(str.maketrans('','',string.punctuation))).split()


def metrics(a,b):
    a,b=words(a),words(b)
    common=sum((Counter(a)&Counter(b)).values())
    return int(a==b),2*common/(len(a)+len(b)) if a or b else 1.


def read(p):
    return json.loads(p.read_text())


def summarize(cases,methods):
    results={}
    rng=random.Random(7913721)
    users=sorted({key[0] for key in cases})
    draws=[rng.choices(users,k=len(users)) for _ in range(5000)]
    for mode in methods:
        obs=[v[mode] for v in cases.values()]
        r={key:mean(x[key] for x in obs) for key in ["cold_s","online_s","setup_s","update_s","search_s",
                 "decoder_positions","encoder_positions","macs","bridge_scalar_ops","logical_old_cache_bytes","skipped_tokens"]}
        r["exact"]=[sum(x["exact"][i] for x in obs) for i in range(2)]
        r["semantic"]=[sum(x["correct"][i] for x in obs) for i in range(2)]
        r["f1"]=[mean(x["f1"][i] for x in obs) for i in range(2)]
        r["correct_both"]=sum(all(x["correct"]) for x in obs)
        r["sequence_matches"]=[sum(z[mode]["tokens_hash"][i]==z["checkpoint"]["tokens_hash"][i] for z in cases.values()) for i in range(2)]
        r["paired"]={}
        for ref in ("fresh","old_rag","notice","checkpoint","splice"):
            diff=[[int(v[mode]["correct"][i])-int(v[ref]["correct"][i]) for v in cases.values()] for i in range(2)]
            r["paired"][ref]={j:[sum(t==val for t in series) for series in diff] for j,val in (("wins",1),("losses",-1))}
        r["ratios"]={}
        for key in ("cold_s","macs"):
            total={u:[sum(v[m][key] for k,v in cases.items() if k[0]==u) for m in (mode,"checkpoint")] for u in users}
            dist=sorted(sum(total[u][0] for u in d)/sum(total[u][1] for u in d) for d in draws)
            r["ratios"][key]=dict(mean=r[key]/mean(v["checkpoint"][key] for v in cases.values()),ci95_users=[dist[124],dist[4874]])
        results[mode]=r
    new=results["bridge"]
    references=("fresh","old_rag","notice","splice")
    acc=all(new["semantic"][0]>=results[m]["semantic"][0]+2 and new["paired"][m]["losses"][1]==0 for m in references)
    fast=new["ratios"]["cold_s"]["mean"]<=.8 and new["ratios"]["macs"]["mean"]<=.8 and sum(new["paired"]["checkpoint"]["losses"])==0
    comp=results["splice"]
    matched=dict(extra_correct=[a-b for a,b in zip(new["semantic"],comp["semantic"])],
                 cold_ratio=new["cold_s"]/comp["cold_s"],compute_ratio=new["macs"]/comp["macs"])
    return dict(episodes=len(cases),users=len(users),policies=results,accuracy_gate=acc,efficiency_gate=fast,matched_comparison=matched)


def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument("data",type=Path);ap.add_argument("public",type=Path)
    a=ap.parse_args()
    parent=a.public/"experiments/halumem_feasibility"
    with a.data.open("rb") as f:raw=[f.readline() for _ in range(4)]
    assert sha256(b''.join(raw)).hexdigest()==read(parent/"audit.json")["input_first_four_digest"]
    users=[json.loads(v) for v in raw]
    rows=list(map(json.loads,(HERE/"rows.jsonl").read_text().splitlines()))
    modes=read(HERE/"contract.json")["methods"]
    expected={tuple(r["key"])+(m,) for r in read(parent/"observations.json") for m in modes}
    assert len(rows)==len(expected) and {tuple(r["key"]) for r in rows}==expected
    prior=read(parent/"blind_grades.json")
    bank={};cases={};review={}
    for r in rows:
        uid,ev,at,m=r["key"]
        entries=[];ex=[];f1=[];tokens_hash=[]
        for index,ans,ts in zip(r["question_indices"],r["replies"],r["tokens"],strict=True):
            q=users[uid]["sessions"][at]["questions"][index]
            material=dict(question=q["question"],reference=q["answer"],response=ans)
            k=sha256(("hmem-g1"+json.dumps(material,sort_keys=True)).encode()).hexdigest()[:12]
            assert k not in bank or bank[k]==material
            bank[k]=material;entries.append(k)
            if k not in prior:review[k]=material
            exact,overlap=metrics(ans,q["answer"]);ex.append(exact);f1.append(overlap)
            tokens_hash.append(sha256(json.dumps(ts,separators=(',',':')).encode()).hexdigest())
        raw={k:v for k,v in r.items() if k not in ("key","tokens","replies")}
        raw.update(exact=ex,f1=f1,blind_ids=entries,tokens_hash=tokens_hash,
                   macs=r["encoder_macs"]+r["decoder_macs"]+r["bridge_scalar_ops"])
        cases.setdefault((uid,ev,at),{})[m]=raw
    (HERE/"blind_new.json").write_text(json.dumps(dict(sorted(review.items())),indent=2)+"\n")
    for key,m in cases.items():
        assert len({str(v["selected"]) for v in m.values()})==1,key
    grade_file=HERE/"grades.json"
    if not grade_file.exists():
        print(f"{len(review)} new responses; {len(bank)-len(review)} prior grades.")
        return
    grades=read(grade_file)
    assert set(grades)==set(review) and all(v in (0,1) for v in grades.values())
    all_scores=prior|grades
    for arms in cases.values():
        for arm in arms.values():arm["correct"]=[all_scores[v] for v in arm["blind_ids"]]
    receipt=[dict(key=k,arms=v) for k,v in sorted(cases.items())]
    (HERE/"observations.json").write_text(json.dumps(receipt,indent=2)+"\n")
    result=summarize(cases,modes)
    result["grading"]=dict(new=len(review),prior=len(bank)-len(review),type="Reference-based method-blind Codex AI review")
    (HERE/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="policies"}))


if __name__=="__main__":
    main()
