"""Recompute sparse-donor calibration and held-out service comparisons."""
import argparse
import json
from pathlib import Path
import random
from statistics import mean, median

HERE=Path(__file__).resolve().parent
MODES=("truncated","joint","absolute","paired")


def score(phase, rows, calibration):
    seeds=list(range(27112000,27112008) if phase=="dev" else range(38456000,38456032))
    expected={(s,f,n,a) for s in seeds for f in ("table","prose") for n in (True,False) for a in (True,False)}
    assert len(rows)==len(expected) and {tuple(r["key"]) for r in rows}==expected
    assert all(set(r["arms"])==set(MODES) for r in rows)
    targets=lambda r:r["key"][2:]==[False,True]
    correct=lambda r,m:r["arms"][m]["top"]==(r["donor"] if targets(r) else r["recipient"])
    stats={}
    for m in MODES:
        controls=sum(correct(r,m) for r in rows if not targets(r))
        stats[m]=dict(targets=sum(correct(r,m) for r in rows if targets(r)),controls=controls,
                      seconds=mean(r["arms"][m]["seconds"] for r in rows),
                      prefix_s=mean(r["arms"][m]["prefix_s"] for r in rows),
                      median_delta_cosine=median(r["arms"][m]["delta_cosine"] for r in rows if r["key"][3]),
                      median_relative_delta_error=median(r["arms"][m]["relative_delta_error"] for r in rows if r["key"][3]))
    rng=random.Random(880137)
    draws=[rng.choices(seeds,k=len(seeds)) for _ in range(5000)]
    def interval(values):return [values[124], values[4874]]
    pairs={}
    for m in MODES[:-1]:
        wins=sum(correct(r,"paired") and not correct(r,m) for r in rows if targets(r))
        losses=sum(not correct(r,"paired") and correct(r,m) for r in rows if targets(r))
        times={s:[sum(r["arms"][j]["seconds"] for r in rows if r["key"][0]==s) for j in ("paired",m)] for s in seeds}
        gains={s:sum(int(correct(r,"paired"))-int(correct(r,m)) for r in rows if r["key"][0]==s and targets(r)) for s in seeds}
        ratios=sorted(sum(times[s][0] for s in draw)/sum(times[s][1] for s in draw) for draw in draws)
        differences=sorted(sum(gains[s] for s in draw)/(2*len(draw)) for draw in draws)
        saved=stats[m]["seconds"]-stats["paired"]["seconds"]
        pairs[m]=dict(wins=wins,losses=losses,time_ratio=stats["paired"]["seconds"]/stats[m]["seconds"],
                      time_ratio_ci95=interval(ratios),accuracy_gain_ci95=interval(differences),
                      calibration_break_even_requests=(int(calibration["elapsed_s"]/saved)+1 if saved>0 and m in ("truncated","joint") else None))
    accuracy=pairs["truncated"]["losses"]==0 and stats["paired"]["controls"]==6*len(seeds)
    speed=all(pairs[m]["time_ratio"]<=.8 for m in ("truncated","joint"))
    return dict(phase=phase,seeds=len(seeds),targets=2*len(seeds),controls=6*len(seeds),methods=stats,paired=pairs,
                promote=phase=="dev" and accuracy and speed and pairs["absolute"]["wins"]-pairs["absolute"]["losses"]>=2,
                confirmed=phase=="test" and accuracy and pairs["absolute"]["accuracy_gain_ci95"][0]>0
                    and all(pairs[m]["time_ratio_ci95"][1]<.8 for m in ("truncated","joint")))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("phase",choices=("dev","test"));phase=ap.parse_args().phase
    s=score(phase,list(map(json.loads,(HERE/(phase+"_rows.jsonl")).read_text().splitlines())),json.loads((HERE/"calibration.json").read_text()))
    (HERE/(phase+"_summary.json")).write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
    print(json.dumps(s))
