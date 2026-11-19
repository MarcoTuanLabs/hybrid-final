"""Score every changed and stable capital, including paired losses on the same questions."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def score(rows):
    assert {tuple(r["key"]) for r in rows}=={(s,f) for s in range(51973000,51973012) for f in ("DEFAULT","WORK")}
    assert len({r["digest"] for r in rows})==len(rows)==24
    modes=("stale","full","notice","swap_values","centered","project","paired","local","recompute")
    assert all(set(r["outputs"])==set(modes) and all(len(v)==3 for v in r["outputs"].values()) for r in rows)
    goals=[(r,i) for r in rows for i in range(3) if i!=r["stable"]]
    stable=[(r,r["stable"]) for r in rows]
    correct=lambda r,i,m:r["outputs"][m][i]==r["new_gold"][i]
    s={}
    for m in modes:
        s[m]=dict(changed=sum(correct(r,i,m) for r,i in goals),stable=sum(correct(r,i,m) for r,i in stable),
              lost_full=sum(correct(r,i,"full") and not correct(r,i,m) for r,i in goals),
              lost_stable=sum(correct(r,i,"stale") and not correct(r,i,m) for r,i in stable),
              gained_full=sum(correct(r,i,m) and not correct(r,i,"full") for r,i in goals),
              wins=sum(correct(r,i,"paired") and not correct(r,i,m) for r,i in goals),
              losses=sum(not correct(r,i,"paired") and correct(r,i,m) for r,i in goals))
    v=s["paired"]
    gate=v["lost_full"]<=2 and v["lost_stable"]==0 and v["changed"]>=s["swap_values"]["changed"]+4
    gate=gate and all(v["changed"]>=s[k]["changed"]+2 for k in ("centered","project"))
    gate=gate and v["changed"]+v["stable"]>=s["recompute"]["changed"]+s["recompute"]["stable"]
    return dict(changed=len(goals),stable=len(stable),documents=len(rows),methods=s,promote=gate)


if __name__=="__main__":
    s=score(list(map(json.loads,(HERE/"dev_rows.jsonl").read_text().splitlines())))
    (HERE/"dev_summary.json").write_text(json.dumps(s,sort_keys=True,indent=2)+"\n");print(json.dumps(s))
