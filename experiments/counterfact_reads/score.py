"""Count free first-token answers by paired subject and original neighborhood query."""
import json
import sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    out=Path(sys.argv[1])
    rows=list(map(json.loads,(out/'rows.jsonl').read_text().splitlines()))
    assert [r["case"] for r in rows]==[c["id"] for c in json.loads((root/"cases.json").read_text())]
    assert len(rows)==len({r["case"] for r in rows})==24
    items=[x for r in rows for x in r["items"]]
    assert all([e["kind"] for e in r["items"]]==["canonical","paraphrase","neighbor"] for r in rows)
    methods=list(items[0]["arms"]);summary={}
    known=sum(e["kind"]=="neighbor" and e["arms"]["old"]["hit"][0] for e in items)
    for m in methods:
        success={kind:0 for kind in ("canonical","paraphrase","neighbor")}
        lost=paired=0
        for e in items:
            assert set(e["arms"])==set(methods)
            stable=e["kind"]=="neighbor";x=e["arms"][m];ix=0 if stable else 1
            success[e["kind"]]+=x["hit"][ix]
            lost+=stable and e["arms"]["old"]["hit"][0] and not x["hit"][0]
            paired+=x["new_minus_old"]<0 if stable else x["new_minus_old"]>0
            if m in ("normalized","matrix","fixed","raw"):assert x["bytes"]==2048
        summary[m]=dict(**success,changed=success["canonical"]+success["paraphrase"],old_correct_stable_lost=lost,pairwise_preference=paired)
    controls=("old","matrix","fixed","raw")
    best=max(summary[m]["changed"] for m in controls);v=summary["normalized"]
    reference=summary["full"]["changed"]-summary["old"]["changed"]
    promote=v["changed"]>=best+4 and v["old_correct_stable_lost"]==0 and reference>0 and 2*(v["changed"]-summary["old"]["changed"])>=reference
    result=dict(cases=len(rows),targets=48,neighbors=24,old_correct_neighbors=known,methods=summary,promote=promote)
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
