"""Pair by seed, retaining both surface forms when testing an objective contrast."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);p.add_argument("phase")
    a=p.parse_args();c=json.loads((HERE/"contract.json").read_text());seeds=c[a.phase+"_seeds"]
    rows=list(map(json.loads,(a.out/"rows.jsonl").read_text().splitlines()))
    assert len(rows)==len(seeds)*8 and {tuple(r["key"]) for r in rows}==set(itertools.product(seeds,("table","prose"),(False,True),(True,False)))
    names=["deployed",*c["objectives"]];assert all(set(r["methods"])==set(names) for r in rows)
    def target(r):return r["key"][3] and not r["key"][2]
    def hit(r,m):return r["methods"][m]["top"]==r["donor" if target(r) else "recipient"]
    m={}
    signs=np.random.default_rng(190871).choice([-1,1],size=(100000,len(seeds)))
    for name in names:
        difference=np.array([sum(int(hit(r,"finite"))-int(hit(r,name)) for r in rows if r["key"][0]==k and target(r)) for k in seeds])
        pvalue=(1+np.count_nonzero(signs@difference>=difference.sum()))/(len(signs)+1)
        m[name]=dict(target=sum(target(r) and hit(r,name) for r in rows),controls=sum(not target(r) and hit(r,name) for r in rows),
                     p_one_sided=float(pvalue),finite_wins=sum(target(r) and hit(r,"finite") and not hit(r,name) for r in rows),
                     finite_losses=sum(target(r) and hit(r,name) and not hit(r,"finite") for r in rows),
                     solver_seconds=sum(r["methods"][name].get("solver_seconds",0) for r in rows) if name!="deployed" else None)
    comparators=c['comparators'];lead=m['finite']['target']-max(m[k]['target'] for k in comparators)
    agreement=sum(r["methods"]["finite"]["top"]==r["methods"]["deployed"]["top"] for r in rows)
    passes=(lead>=(4 if a.phase=="confirm" else 2) and m['finite']['controls']==6*len(seeds) and agreement==len(rows))
    if a.phase=="confirm":passes=passes and all(m[k]['p_one_sided']<=0.05/len(comparators) for k in comparators)
    s=dict(phase=a.phase,targets=2*len(seeds),controls=6*len(seeds),methods=m,finite_agrees_with_deployed=agreement,
           advantage_over_best=lead,pass_gate=passes)
    (a.out/"summary.json").write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s))
if __name__=="__main__":main()
