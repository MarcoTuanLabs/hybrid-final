"""Recompute unconditional control and paired comparisons on all registered CAPITALS prompts."""
import argparse
import json
from pathlib import Path
import random
from statistics import median

HERE=Path(__file__).resolve().parent


def score(phase,rows):
    seeds=list(range(51973000,51973012) if phase=="dev" else range(51974000,51974064))
    assert {tuple(r["key"]) for r in rows}=={(s,f,n) for s in seeds for f in ("DEFAULT","WORK") for n in (False,True)}
    assert len({r["source_pair"] for r in rows})==len(rows)==4*len(seeds)
    modes=("recipient","donor","native","wide_native","white","normal","white_unwritten","normal_unwritten")
    assert all(set(r["outputs"])==set(modes) for r in rows)
    goals=[r for r in rows if not r["key"][2]]
    stable=[r for r in rows if r["key"][2]]
    def token(r,m):return r["outputs"][m]["top"]
    def win(r,m):return token(r,m)==r["alternate"]
    clean=[r for r in goals if token(r,"recipient")==r["recipient"] and token(r,"donor")==r["alternate"]]
    per={}
    for m in modes:
        per[m]=dict(target=sum(win(r,m) for r in goals),nuisance_correct=sum(token(r,m)==r["recipient"] for r in stable),
                    nuisance_changed=sum(token(r,m)!=token(r,"recipient") for r in stable),
                    target_on_clean=sum(win(r,m) for r in clean),target_margin_median=median(r["outputs"][m]["margin"] for r in goals))
    rng=random.Random(72985019);draws=[rng.choices(seeds,k=len(seeds)) for _ in range(5000)]
    paired={}
    for m in ("white","native","wide_native"):
        wins=sum(win(r,"normal") and not win(r,m) for r in goals)
        losses=sum(not win(r,"normal") and win(r,m) for r in goals)
        diffs={s:sum(int(win(r,"normal"))-int(win(r,m)) for r in goals if r["key"][0]==s) for s in seeds}
        v=sorted(sum(diffs[s] for s in sample)/(2*len(seeds)) for sample in draws)
        paired[m]=dict(wins=wins,losses=losses,gain=(wins-losses)/len(goals),ci95=[v[124],v[4874]])
    controls_changed=sum(token(r,"normal_unwritten")!=token(r,"recipient") for r in rows)+per["normal"]["nuisance_changed"]
    gains={m:per["normal"]["target"]-per[m]["target"] for m in ("native","white")}
    return dict(phase=phase,seeds=len(seeds),targets=len(goals),controls=len(goals)*3,clean_pairs=len(clean),
                clean_recipient_target=sum(token(r,"recipient")==r["recipient"] for r in goals),
                methods=per,paired=paired,changed_control_outputs=controls_changed,wide_coordinates=list({r["wide_coordinates"] for r in rows}),
                promote=phase=="dev" and len(clean)>=18 and min(gains.values())>=4 and controls_changed==0,
                confirmed=phase=="test" and min(gains.values())>=10 and all(paired[m]["ci95"][0]>0 for m in gains) and controls_changed==0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("phase",choices=("dev","test"));phase=p.parse_args().phase
    rows=list(map(json.loads,(HERE/(phase+"_rows.jsonl")).read_text().splitlines()))
    s=score(phase,rows);(HERE/(phase+"_summary.json")).write_text(json.dumps(s,sort_keys=True,indent=2)+"\n")
    print(json.dumps(s))
