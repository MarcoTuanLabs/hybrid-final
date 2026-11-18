"""Count verbatim changed-answer matches in raw dialogue without selecting episodes or answers."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

HERE=Path(__file__).resolve().parent


def normalize(text):
    ws=re.findall(r"\w+",unicodedata.normalize("NFKD",text).casefold().replace("_"," "))
    return " "+" ".join(ws)+" "


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    args=parser.parse_args()
    with args.data.open("rb") as f: users=[json.loads(f.readline()) for _ in range(4)]
    rows=json.loads((HERE/"observations.json").read_text())
    out=[]
    for r in rows:
        u,event,at=r["key"]
        sessions=users[u]["sessions"]
        q=sessions[at]["questions"][r["question_indices"][0]]
        gold=normalize(q["answer"])
        matched={i for i,s in enumerate(sessions[:at+1])
                 if any(gold in normalize(turn["content"]) for turn in s.get("dialogue",[]))}
        out.append(dict(key=r["key"],in_event=event in matched,in_old=bool(matched-{event}),
                        in_updated=bool(matched)))
    result=dict(changed_questions=len(out),per_episode=out,
                counts={k:sum(x[k] for x in out) for k in ["in_event","in_old","in_updated"]})
    (HERE/"grounding.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result["counts"])


if __name__=="__main__":
    main()
