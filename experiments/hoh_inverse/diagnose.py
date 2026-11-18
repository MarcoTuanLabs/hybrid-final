"""Score revision relevance without converting a source-ranking difference into a QA claim."""
import json
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'hoh_updates'
def mean(xs):return statistics.mean(xs)
def main():
    labels=json.loads((SOURCE/'labels.json').read_text())
    rows=list(map(json.loads,(HERE/'likelihoods.jsonl').read_text().splitlines()))
    assert set(labels)=={r['id'] for r in rows} and len(rows)==32
    rng=random.Random(124678);draws=[rng.choices(range(len(rows)),k=len(rows)) for _ in range(4000)]
    report={};group={}
    for mode in ['upr','paired','whole_gain','recurrent_gain']:
        units=[];maxs=[];pos=[];neg=[];top=0
        for row in rows:
            key=labels[row['id']]['source_key'];scores=row['scores']
            a=scores[key][mode];other=[v[mode] for k,v in scores.items() if k!=key]
            assert len(other)==5
            units.append(mean([1. if a>x else .5 if a==x else 0. for x in other]))
            pos.append(a);neg.append(mean(other));maxs.append(max(abs(v[mode]) for v in scores.values()))
            top+=a>max(other)
        boots=sorted(mean([units[i] for i in sample]) for sample in draws)
        group[mode]=units
        report[mode]=dict(auc=mean(units),page_paired_auc_ci=[boots[100],boots[3899]],strict_top1=top,
             median_source=statistics.median(pos),median_per_query_others=statistics.median(neg),median_max_absolute_score=statistics.median(maxs))
    diffs={}
    for pair in [('paired','upr'),('recurrent_gain','whole_gain')]:
        diff=[x-y for x,y in zip(*(group[k] for k in pair),strict=True)]
        bs=sorted(mean([diff[i] for i in sample]) for sample in draws)
        diffs[' minus '.join(pair)]=dict(auc_difference=mean(diff),ci95=[bs[100],bs[3899]])
    (HERE/'relevance.json').write_text(json.dumps(dict(n=len(rows),features=report,paired_differences=diffs),indent=2)+'\n')
if __name__=='__main__':main()
