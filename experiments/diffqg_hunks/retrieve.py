"""Build lexical and neural indexes from plain revision passages, not annotated answer spans."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from search import Encoder,BM25,words,reciprocal

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    start=time.monotonic();bank=json.loads((HERE/'bank.json').read_text())
    qs=json.loads((HERE/'questions.json').read_text());enc=Encoder(a.model)
    text=[r['title']+' — '+r['current'] for r in bank]
    old=[r['title']+' — '+r['old'] for r in bank]
    bj=BM25(text);bo=BM25(old)
    # Dense text is independent of all user questions.
    vec=enc.encode(text)
    qvec=enc.encode([r['question'] for r in qs])
    scores=qvec@vec.T;index_s=time.monotonic()-start;out={}
    for j,q in enumerate(qs):
        terms=dict.fromkeys(words(q['question']),1.)
        lexical=bj.search(terms,len(bank));prior=bo.search(terms,len(bank))
        dense=np.argsort(-scores[j],kind='stable').tolist()
        def combine(eligible):
            orders=[[k for k in order if k in eligible] for order in (lexical,prior,dense)]
            vecs=[reciprocal(x) for x in orders]
            return sorted(sorted(eligible),key=lambda i:-(vecs[0].get(i,0)+.35*vecs[1].get(i,0)+vecs[2].get(i,0)))
        valid={i for i,row in enumerate(bank) if row['date']==q['as_of']}
        all_years=combine(set(range(len(bank))))
        merged=combine(valid)
        out[q['id']]=dict(main=[bank[i]['key'] for i in merged[:6]],all_years=[bank[i]['key'] for i in all_years[:6]],
                          lexical=[bank[i]['key'] for i in lexical if i in valid][:6],
                          dense=[bank[i]['key'] for i in dense if i in valid][:6])
        assert out[q['id']]['main']
    (HERE/'ranks.json').write_text(json.dumps(out)+'\n')
    print(json.dumps(dict(n=len(bank),questions=len(qs),index_s=index_s,seconds=time.monotonic()-start,
                   question_sha256=hashlib.sha256((HERE/'questions.json').read_bytes()).hexdigest())))
if __name__=='__main__':main()
