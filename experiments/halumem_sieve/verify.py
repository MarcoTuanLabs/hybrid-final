"""Check coverage against original turn indices and whole-turn budget receipts."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    def read(n):return json.loads((HERE/n).read_text())
    for n,h in read("bundle.json")["sha256"].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,n
    s=read("summary.json");sel=read("selections.json");qs=read("witnesses.json")
    assert len(sel)==s["sessions"] and len(qs)==s["questions"]
    assert sum(bool(v["witnesses"]) for v in qs.values())==s["attainable"]
    assert sum(v["total"] for v in sel.values())==s["original_tokens"]
    cover={}
    for m,summary in s["methods"].items():
        cover[m]={j for j,v in qs.items() if set(v["witnesses"]) & set(sel[v["event"]]["methods"][m]["indices"])}
        assert len(cover[m])==summary["covered"]
        assert sum(v["methods"][m]["tokens"] for v in sel.values())==summary["selected_tokens"]
        assert all(v["methods"][m]["tokens"]<=v["budget"]<=v["total"]/4 for v in sel.values())
    for m,summary in s["methods"].items():
        assert summary["normalized_wins"]==len(cover["normalized"]-cover[m])
        assert summary["normalized_losses"]==len(cover[m]-cover["normalized"])
    best=max(len(v) for m,v in cover.items() if m not in ("raw","normalized"))
    n=len(cover["normalized"])
    assert s["promote"]==(s["attainable"]>=8 and s["eligible_users"]>=3 and n>=.8*s["attainable"] and n>=best+2 and n>=len(cover["raw"])+1)
    print("Verified whole-turn budget and all paired coverage scores.")


if __name__=="__main__":main()
