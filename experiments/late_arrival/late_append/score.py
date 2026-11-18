"""Reuse the existing LongMemEval lexical scoring rule and compare every policy."""
import json
import re
from collections import Counter
from pathlib import Path

P=Path(__file__).resolve().parent


def words(s):
    return re.findall(r'\w+',str(s).casefold())


def token_f1(a,b):
    x,y=words(a),words(b)
    return 2*sum((Counter(x)&Counter(y)).values())/(len(x)+len(y)) if x or y else 1.


def main():
    labels=json.loads((P.parent/'suffix_transport/labels.json').read_text())
    rows=list(map(json.loads,(P/'rows.jsonl').read_text().splitlines()))
    methods=json.loads((P/'contract.json').read_text())['modes']
    assert len(rows)==len(labels)*len(methods)==len({(r['id'],r['mode']) for r in rows})
    reference={r['id']:r for r in rows if r['mode']=='full'}
    stats={}
    for mode in methods:
        sub=[r for r in rows if r['mode']==mode]
        stats[mode]=dict(exact=sum(words(r['text'])==words(labels[r['id']]) for r in sub),
                         token_f1=sum(token_f1(r['text'],labels[r['id']]) for r in sub)/len(sub),
                         exact_sequences=sum(r['seq']==reference[r['id']]['seq'] for r in sub),
                         positions=sum(r['positions'] for r in sub),
                         lifecycle_s=sum(r['ingest_s']+r['update_answer_s'] for r in sub))
    full,gates=stats['full'],stats['gates']
    passed=(gates['exact']>=full['exact'] and gates['token_f1']>=full['token_f1']
            and gates['positions']<=.8*full['positions']
            and gates['exact']>=max(stats[m]['exact'] for m in ('append','dated_append'))+2)
    result=dict(n=len(labels),stats=stats,gate_passed=passed)
    (P/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    main()
