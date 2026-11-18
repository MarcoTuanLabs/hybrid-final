"""Independently replay all six donor orders for each old/new snapshot comparison."""
import argparse
from itertools import permutations
from pathlib import Path
from statistics import mean,median

from check import HERE,S,lines,read,marks

D=HERE/'asof_attribution'
NAMES=('attention','convolution','recurrent')


def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:close(a[key],b[key])
    else:
        assert abs(a-b)<1e-7,(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=['4b','9b'])
    parser.add_argument('--rows',type=Path)
    args=parser.parse_args()
    rows=lines(args.rows or D/(args.model+'_rows.jsonl'))
    gold=read(S/'labels.json');prior=read(S/'4b_summary.json')['per_page']
    assert len(rows)==len(gold) and {r['id'] for r in rows}==gold.keys()
    ref=read(D/(args.model+'_summary.json'))
    for r in rows:
        scores=r['log_scores'];odds={k:ll[1]-ll[0] for k,ll in scores.items()};close(odds,r['log_odds'])
        acc=[[] for _ in NAMES]
        for order in permutations(range(3)):
            current=['0','0','0']
            for j in order:
                before=odds[''.join(current)];current[j]='1';acc[j].append(odds[''.join(current)]-before)
        close({name:mean(v) for name,v in zip(NAMES,acc,strict=True)},r['contribution'])
        close(odds['111']-odds['000'],r['total_gap']);close(sum(r['contribution'].values()),r['total_gap'])
        assert r['identity_max_error']<1e-5
    missed=[r for r in rows if not all(prior[r['id']]['all'])]
    correct=[r for r in rows if all(prior[r['id']]['all'])]
    for group,rs in [('all',rows),('prior_missed',missed),('prior_correct',correct)]:
        expected=ref['groups'][group]
        detail={}
        for name in NAMES:
            vals=[r['contribution'][name] for r in rs]
            detail[name]=dict(positive_and_total_above_threshold=sum(v>.05 and r['total_gap']>.05 for v,r in zip(vals,rs,strict=True)),
                median=median(vals),mean=mean(vals),largest_contributor=sum(name==max(r['contribution'],key=r['contribution'].get) for r in rs))
        greedy=[[marks(r['open_answers'][tag],str(gold[r['id']][i]))[0] for i,tag in enumerate(('000','111'))] for r in rs]
        actual=dict(pages=len(rs),gap_above_threshold=sum(r['total_gap']>.05 for r in rs),
             favorable_assignment=sum(r['total_gap']>0 for r in rs),both_forced_choices_correct=sum(r['log_odds']['000']<0<r['log_odds']['111'] for r in rs),
             old_forced_correct=sum(r['log_odds']['000']<0 for r in rs),new_forced_correct=sum(r['log_odds']['111']>0 for r in rs),
             greedy_both=sum(all(pair) for pair in greedy),greedy_individual=sum(sum(pair) for pair in greedy),
             median_total=median(r['total_gap'] for r in rs),channels=detail)
        close(actual,expected)
    assert ref['diagnostic_gate']==(ref['groups']['prior_missed']['channels']['recurrent']['positive_and_total_above_threshold']>=4)
    print(args.model,'counterfactual scores verified on',len(rows),'pages and',8*len(rows),'cache configurations.')


if __name__=='__main__':
    main()
