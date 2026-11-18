"""Separate raw candidate turns from evaluation-only source witnesses and references."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent


def norm(s):
    return " "+" ".join(re.findall(r"\w+",unicodedata.normalize("NFKD",s).casefold().replace("_"," ")))+" "


def main():
    p=argparse.ArgumentParser();p.add_argument("data",type=Path);p.add_argument("model",type=Path);p.add_argument("public",type=Path);a=p.parse_args()
    audit=json.loads((a.public/"experiments/halumem_feasibility/audit.json").read_text())
    with a.data.open("rb") as f:raw=[f.readline() for _ in range(4)]
    assert sha256(b"".join(raw)).hexdigest()==audit["input_first_four_digest"]
    users=[json.loads(line) for line in raw];tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    inputs={};targets={}
    for e in audit["candidate_pairs"]:
        if not e["unaffected_evidence_elsewhere"] or e["intervening_tokens"]<128:continue
        u,i,at=e["user"],e["event"],e["question_session"];k=f"{u}/{i}";s=users[u]["sessions"]
        def fmt(t):return f"{t['timestamp']} | {t['role']} | {t['content']}\n"
        if k not in inputs:
            turns=[dict(id=n,role=t["role"],text=fmt(t),tokens=len(tok.encode(fmt(t),add_special_tokens=False)))
                   for n,t in enumerate(s[i]["dialogue"])]
            inputs[k]=dict(prior="".join(fmt(t) for t in s[i-1]["dialogue"]),turns=turns,budget=sum(t["tokens"] for t in turns)//4)
        for j in e["changed"]:
            key=f"{u}/{at}/{j}"
            q=s[at]["questions"][j];gold=norm(q["answer"])
            witnesses=[n for n,t in enumerate(s[i]["dialogue"]) if gold in norm(t["content"])]
            assert key not in targets
            targets[key]=dict(event=k,witnesses=witnesses,user=u)
    (HERE/"raw_inputs.json").write_text(json.dumps(inputs,ensure_ascii=False)+"\n")
    (HERE/"witnesses.json").write_text(json.dumps(targets,indent=2)+"\n")
    print(len(inputs),"sessions,",len(targets),"questions,",sum(bool(x["witnesses"]) for x in targets.values()),"with literal matches")


if __name__=="__main__":main()
