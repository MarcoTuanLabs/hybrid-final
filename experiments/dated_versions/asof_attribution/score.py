"""Summarize paired candidate odds and the three exact cache-component contributions."""
import argparse
import json
from pathlib import Path
from statistics import mean,median
import sys

P=Path(__file__).resolve().parent;R=P.parent
sys.path.insert(0,str(R/'asof_revision'))
from score import load_helpers,mark


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('official',type=Path);ap.add_argument('tag');a=ap.parse_args()
    read=lambda f:json.loads(f.read_text())
    baseline=read(R/'asof_revision'/('4b_summary.json'))['per_page']
    gold=read(R/'asof_revision/labels.json')
    items={r['id']:r for r in read(R/'asof_revision/ranked.json')}
    helpers=load_helpers(a.official/'official_utils.py',a.official/'evaluate.py')
    mapping=read(a.official/'party_mapping.json')
    rows=list(map(json.loads,(P/(a.tag+'_rows.jsonl')).read_text().splitlines()))
    assert len(rows)==len({r['id'] for r in rows})==len(gold) and {r['id'] for r in rows}==gold.keys()
    groups={'all':rows,'prior_missed':[r for r in rows if not all(baseline[r['id']]['all'])],
            'prior_correct':[r for r in rows if all(baseline[r['id']]['all'])]}
    result={}
    for name,rs in groups.items():
        channels={}
        for c in ('attention','convolution','recurrent'):
            vals=[r['contribution'][c] for r in rs]
            channels[c]=dict(positive_and_total_above_threshold=sum(r['total_gap']>.05 and v>.05 for r,v in zip(rs,vals,strict=True)),
                             median=median(vals),mean=mean(vals),largest_contributor=sum(c==max(r['contribution'],key=r['contribution'].get) for r in rs))
        greedy=[[mark(r['open_answers'][bit],str(gold[r['id']][j]),items[r['id']],helpers,mapping)[0] for j,bit in enumerate(('000','111'))] for r in rs]
        result[name]=dict(pages=len(rs),gap_above_threshold=sum(r['total_gap']>.05 for r in rs),
             favorable_assignment=sum(r['total_gap']>0 for r in rs),both_forced_choices_correct=sum(r['log_odds']['000']<0<r['log_odds']['111'] for r in rs),
             old_forced_correct=sum(r['log_odds']['000']<0 for r in rs),new_forced_correct=sum(r['log_odds']['111']>0 for r in rs),
             greedy_both=sum(all(pair) for pair in greedy),greedy_individual=sum(sum(pair) for pair in greedy),
             median_total=median(r['total_gap'] for r in rs),channels=channels)
    counts=result['prior_missed']['channels']
    out=dict(groups=result,threshold=.05,diagnostic_gate=counts['recurrent']['positive_and_total_above_threshold']>=4,
             max_identity_error=max(r['identity_max_error'] for r in rows),
             max_efficiency_error=max(abs(r['total_gap']-sum(r['contribution'].values())) for r in rows))
    (P/(a.tag+'_summary.json')).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))
if __name__=='__main__':main()
