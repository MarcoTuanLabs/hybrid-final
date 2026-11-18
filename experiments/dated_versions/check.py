"""Rescore both published studies, including every answer and old/new pair."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import re
import string
import unicodedata

HERE=Path(__file__).resolve().parent
S=HERE/'asof_revision'
T=HERE/'revision_pairs'


def read(path):
    if path.suffix=='.gz':return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def lines(path):
    return list(map(json.loads,path.read_text().splitlines()))


def normalize(s,official):
    if official:
        s=unicodedata.normalize('NFD',s.replace('-',' ')).encode('ascii','ignore').decode()
        s=re.sub(r'[^a-zA-Z0-9\s]','',s)
    s=s.lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())


def marks(answer,reference):
    plain=normalize(answer,False)==normalize(reference,False)
    for marker in ('Provide the shortest answer','||\n\n## Step','||.','|||',' |'):
        answer=answer.split(marker)[0].strip()
    a,b=[normalize(x,True).split() for x in (answer,reference)]
    common=sum((Counter(a)&Counter(b)).values())
    f1=(2*common/(len(a)+len(b))) if a and b else float(a==b)
    return int(a==b),int(plain),f1


def summary(pred,gold):
    stats=defaultdict(Counter);flags={}
    for uid,methods in pred.items():
        flags[uid]={}
        for name,pair in methods.items():
            marked=[marks(ans,ref) for ans,ref in zip(pair,gold[uid],strict=True)]
            a,b=[v[0] for v in marked];flags[uid][name]=[a,b]
            c=stats[name];c['old']+=a;c['new']+=b;c['both']+=a*b;c['correct']+=a+b
            c['plain_exact']+=sum(v[1] for v in marked);c['sum_f1']+=sum(v[2] for v in marked)
    for v in stats.values():v['mean_f1']=v.pop('sum_f1')/(2*len(pred))
    return stats,flags


def verify_stats(actual,expected):
    assert actual.keys()==expected.keys()
    for method,counts in actual.items():
        assert counts.keys()==expected[method].keys()
        for key,value in counts.items():
            assert abs(value-expected[method][key])<1e-9,(method,key,value,expected[method][key])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--erase-rows',type=Path,default=S/'4b_rows.jsonl')
    ap.add_argument('--pair-rows',type=Path,default=T/'4b_rows.jsonl')
    args=ap.parse_args();gold=read(S/'labels.json')
    items={r['id']:r for r in read(S/'ranked.json.gz')};span=read(S/'span_answers.json')
    assert items.keys()==gold.keys()==span.keys() and len(gold)==20
    for item in items.values():
        assert all(k not in item['question'] for k in ('party of',' year ',' years ',' date ',' dates ','term start',' period '))
    erows=lines(args.erase_rows);prows=lines(args.pair_rows)
    for rows in (erows,prows):assert len(rows)==len(gold) and {r['id'] for r in rows}==gold.keys()
    pred={}
    for r in erows:
        uid=r['id'];answers=r['answers']
        pairs={k:[answers[str(i)][k] for i in (0,1)] for k in answers['0']}
        pairs['extractive_span']=span[uid]
        top=[ev[0] for ev in items[uid]['evidence']]
        pairs['field']=[t['cells'][-1] if len(t['cells'])>=2 else t['text'] for t in top];pred[uid]=pairs
    stats,flags=summary(pred,gold);base=read(S/'4b_summary.json')
    verify_stats(stats,base['stats'])
    assert flags==base['per_page']
    pred={r['id']:{**r['proposals'],**{'score_'+k:v for k,v in r['winners'].items()}} for r in prows}
    stats2,_=summary(pred,gold);res=read(T/'4b_summary.json');verify_stats(stats2,res['stats'])
    ceilings=[[[marks(x,s)[0] for x,s in zip(pair,gold[r['id']],strict=True)] for pair in r['options']] for r in prows]
    assert sum(any(all(v) for v in pools) for pools in ceilings)==res['eligible_pool_ceiling_pairs']
    assert sum(sum(any(v[i] for v in pools) for i in (0,1)) for pools in ceilings)==res['eligible_pool_ceiling_individual']
    for target,controls,passed in [(stats['attention_skip'],[v for k,v in stats.items() if k!='attention_skip'],base['gate_passed']),
                                 (stats2['score_tangent'],[*stats.values(),*(v for k,v in stats2.items() if k!='score_tangent')],res['gate_passed'])]:
        assert all(target['both']>=v['both']+3 and target['old']>=v['old'] and target['new']>=v['new'] for v in controls)==passed
    print('Both snapshot experiments verified on all',2*len(gold),'dated questions.')


if __name__=='__main__':
    main()
