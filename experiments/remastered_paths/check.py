"""Rescore the archived predictions and estimate paired differences within source cases."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'walk_transfer'))
from score import normalize,score


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return list(map(json.loads,path.read_text().splitlines()))


def compare(a,b,labels):
    grouped=defaultdict(list)
    won=lost=0
    for uid,key in labels.items():
        aliases=set(map(normalize,key['aliases']))
        outcome=lambda answers:bool(answers[uid].strip()) and normalize(answers[uid]) in aliases
        delta=int(outcome(a))-int(outcome(b))
        grouped[(key['bank'],key['case'])].append(delta)
        won+=delta>0
        lost+=delta<0
    assert len(grouped)==48 and all(len(vals)==2 for vals in grouped.values())
    groups=[sum(pair) for pair in grouped.values()]
    rng=random.Random(20261118)
    samples=sorted(sum(rng.choices(groups,k=len(groups)))/len(labels) for _ in range(10000))
    return dict(won=won,lost=lost,accuracy_difference=(won-lost)/len(labels),paired_case_bootstrap_95=[samples[249],samples[9749]])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,help='write a summary here; otherwise compare against the release')
    ap.add_argument('--runs',type=Path,default=HERE,help='root for archived or fresh walk outputs')
    args=ap.parse_args()
    keys=read(HERE.parent/'remastered_contrast/labels.json')
    path=args.runs
    result={}
    for tag in ('4b','9b'):
        rows=lines(path/'walk_transfer'/(tag+'_rows.jsonl'))
        semantic={r['id']:r for r in lines(path/'walk_transfer'/(tag+'_semantic_rows.jsonl'))}
        assert len(semantic)==96 and set(semantic)==set(keys)
        for row in rows:
            row['outputs']['semantic_gwalk']={'answer':semantic[row['id']]['answer']}
        if tag=='4b':
            paired=lines(path/'paired_hops/4b_rows.jsonl')
            assert len(paired)==len({x['id'] for x in paired})==96
            by_id={r['id']:r for r in paired}
            for row in rows:
                other=by_id[row['id']]
                assert other['plain_replication'] and other['outputs']['plain']==row['outputs']['lexical']['answer']
                for method,answer in other['outputs'].items():
                    row['outputs']['paired_'+method]={'answer':answer}
            result['4b_instantiations']=dict(eligible_questions=sum(bool(r['contrasts']) for r in paired),
                                              distinct_within_question=sum(len(r['contrasts']) for r in paired))
        out=score(rows,keys)
        pred={name:{r['id']:r['outputs'][name]['answer'] for r in rows} for name in rows[0]['outputs']}
        pred['ranked']={r['id']:r['reference'] for r in rows}
        out['paired_case_resampling']={name:compare(pred['lexical'],pred[name],keys) for name in ('ranked','residual_read','semantic_gwalk','transport')}
        result[tag]=out
    if args.runs==HERE:
        aligned=lines(HERE/'clause_reads/4b_qa_rows.jsonl')
        for row in aligned:
            row['outputs']={method+'_'+policy:{'answer':output[policy]}
                            for method,output in row['outputs'].items() for policy in ('walk','read')}
        aligned_counts=score(aligned,keys)['stats']
        expected=read(HERE/'clause_reads/4b_qa_summary.json')['stats']
        assert aligned_counts.pop('ranked')==expected['adaptive']
        assert aligned_counts=={mode:counts for mode,counts in expected.items() if mode!='adaptive'}
    for size in ('4b','9b'):
        stats=result[size]['stats']
        print(size,', '.join(k+' '+str(s['total'])+'/96' for k,s in stats.items()))
    if args.out:
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    else:
        assert result==read(HERE/'summary.json'), 'Replayed scores differ from the released summary.'
        print('Archived counts and paired intervals verified.')


if __name__=='__main__':
    main()
