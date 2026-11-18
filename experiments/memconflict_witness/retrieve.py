"""Rank dated conversational evidence without profile, query category or episode annotations."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from search import BM25,Encoder,words,reciprocal
def segments(history):
    rows=[]
    for s in history:
        # Each record contains two adjacent user/assistant exchanges. References are local to a persona.
        for start in range(0,len(s['turns']),4):
            turns=s['turns'][start:start+4]
            rows.append(dict(session=s['session'],date=s['date'],offset=start,turns=turns))
    return rows
def text(row):return ' '.join(v['text'] for v in row['turns'])
def unique(items):return list(dict.fromkeys(items))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',type=Path);a=parser.parse_args()
    read=lambda n:json.loads((HERE/(n+'.json')).read_text())
    qs=read('questions');hs=read('histories');rw=read('expansions');encoder=Encoder(a.model)
    output={};begin=time.monotonic()
    for q in qs:
        uid=q['id'];docs=segments(hs[uid]);texts=[text(d) for d in docs];bm=BM25(texts)
        terms=dict.fromkeys(words(q['question']),1.)
        expanded=dict.fromkeys(words(rw[uid]['rewrite']),.55)|terms
        base=bm.search(terms,len(docs));broad=bm.search(expanded,len(docs))
        by_session={}
        for j in broad+base:by_session.setdefault(docs[j]['session'],j)
        # Include a composition-independent retrieval option in the actual query session.
        current=docs[-1]['session']
        recent=by_session.get(current,len(docs)-1)
        pool=unique(base[:40]+broad[:40]+list(by_session.values())+[recent])
        v=encoder.encode([q['question']])[0];d=encoder.encode([texts[j] for j in pool])@v
        sem=[pool[j] for j in np.argsort(-d,kind='stable')]
        terms=[reciprocal(k) for k in [base,broad,sem]]
        ranked=sorted(pool,key=lambda j:-sum(t.get(j,0) for t in terms))
        per_session={}
        for j in ranked:per_session.setdefault(docs[j]['session'],j)
        different=list(per_session.values());distinct=unique(different+ranked)[:6]
        leaders=different[:12];ends=sorted(leaders,key=lambda j:docs[j]['session'])
        witnesses=unique([different[0],ends[0],ends[-1],recent]+different+ranked)[:6]
        selections={'standard':ranked[:6],'diverse':distinct,'witness':witnesses,'wide':unique(different+ranked)[:12]}
        output[uid]={k:[docs[j] for j in sorted(ids,key=lambda j:(docs[j]['session'],docs[j]['offset']))] for k,ids in selections.items()}
    (HERE/'evidence.json').write_text(json.dumps(output,ensure_ascii=False)+'\n')
    (HERE/'rank_receipt.json').write_text(json.dumps(dict(questions=len(qs),seconds=time.monotonic()-begin))+'\n')
if __name__=='__main__':main()
