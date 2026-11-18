"""Score actual free top tokens, controls and clustered paired differences."""
import argparse
import json
import math
from pathlib import Path
import random
from statistics import median

HERE=Path(__file__).resolve().parent
CURVE=("ccq_half","ccq_one","ccq_write","key_cov_white")


def score(phase,rows,modes):
    seeds=list(range(27112000,27112016) if phase=="dev" else range(72984000,72984032))
    expected={(s,style,n) for s in seeds for style in ("table","prose") for n in (False,True)}
    assert {tuple(v["key"]) for v in rows}==expected and len(rows)==len(expected)
    assert all(set(v["methods"])==set(modes) and len(v["slots"])==4 for v in rows)
    n=len(seeds)*2;stats={};pair={}
    rng=random.Random(791782);draws=[rng.choices(seeds,k=len(seeds)) for _ in range(5000)]
    target=lambda v,m:v["methods"][m]["top"][1]==v["donor"]
    for m in modes:
        goals=[v for v in rows if not v["key"][2]]
        null=[v for v in rows if v["key"][2]]
        angles=[z["angles"][m] for v in rows for z in v["slots"]]
        assert all(math.isfinite(x) for x in angles)
        stats[m]=dict(target=sum(target(v,m) for v in goals),
                      written_nuisance=sum(v["methods"][m]["top"][1]==v["recipient"] for v in null),
                      unwritten_target=sum(v["methods"][m]["top"][0]==v["recipient"] for v in goals),
                      unwritten_nuisance=sum(v["methods"][m]["top"][0]==v["recipient"] for v in null),
                      median_degrees=median(angles),max_degrees=max(angles))
        diffs={seed:sum(int(target(v,"normalized"))-int(target(v,m)) for v in goals if v["key"][0]==seed) for seed in seeds}
        distribution=sorted(sum(diffs[i] for i in d)/n for d in draws)
        pair[m]=dict(wins=sum(target(v,"normalized") and not target(v,m) for v in goals),
                     losses=sum(target(v,m) and not target(v,"normalized") for v in goals),
                     gain=sum(diffs.values())/n,ci95=[distribution[124],distribution[4874]])
    v=stats["normalized"];preserved=all(v[g]==n for g in ("written_nuisance","unwritten_target","unwritten_nuisance"))
    better=v["target"]-max(stats[m]["target"] for m in CURVE)
    return dict(phase=phase,seeds=len(seeds),targets=n,control_presentations=3*n,methods=stats,paired_normalized=pair,
                lambda_histogram={str(l):sum(z["chosen_lambda"]==l for v in rows for z in v["slots"]) for l in (0,.25,.5,.75,1)},
                development_pass=phase=="dev" and preserved and better>=4,
                confirmation_pass=phase=="test" and preserved and better>=8 and all(pair[m]["ci95"][0]>0 for m in CURVE))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("phase",choices=("dev","test"));phase=ap.parse_args().phase
    rows=list(map(json.loads,(HERE/(phase+"_rows.jsonl")).read_text().splitlines()))
    stats=score(phase,rows,json.loads((HERE/"contract.json").read_text())["policies"])
    (HERE/(phase+"_summary.json")).write_text(json.dumps(stats,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in stats.items() if k not in ("methods","paired_normalized","lambda_histogram")}))
    print({m:v["target"] for m,v in stats["methods"].items()})


if __name__=="__main__":main()
