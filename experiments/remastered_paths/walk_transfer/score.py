"""Exact answer scores and paired differences for the same questions at both reader sizes."""
import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path


def normalize(answer):
    clean = answer.casefold().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', clean).split())


def score(rows, labels):
    assert len(rows)==len(labels)==len({r['id'] for r in rows})
    stats=defaultdict(Counter)
    correct=defaultdict(set)
    for row in rows:
        uid=row['id']
        item=labels[uid]
        key=item['role']+'_'+item['structure']
        accepted={normalize(x) for x in item['aliases']}
        predictions={method:value['answer'] for method,value in row['outputs'].items()}
        predictions['ranked']=row['reference']
        for method,answer in predictions.items():
            ok=bool(answer.strip()) and normalize(answer) in accepted
            stats[method][key]+=int(ok)
            stats[method]['total']+=int(ok)
            if ok:
                correct[method].add(uid)
    reference=correct.get('lexical',set())
    peers={name:dict(won=len(ids-reference),lost=len(reference-ids)) for name,ids in correct.items()} if 'lexical' in correct else {}
    return dict(n=len(rows),stats=stats,paired_against_lexical=peers)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rows',type=Path);ap.add_argument('labels',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args()
    rows=list(map(json.loads,args.rows.read_text().splitlines()))
    result=score(rows,json.loads(args.labels.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v['total'] for k,v in result['stats'].items()})

if __name__=='__main__':
    main()
