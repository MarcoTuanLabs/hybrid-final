"""Answer open queries from the same candidate sources under each frozen ranking rule."""
import argparse
import json
from pathlib import Path
import time
import mlx.core as mx
from query import Engine,generate,SOURCE

HERE=Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text())
def rows(path,field):
    return {x[field]:x for x in map(json.loads,path.read_text().splitlines())}
def ranks(keys,values):
    order=sorted(keys,key=values.get,reverse=True)
    return {k:1/(12+j) for j,k in enumerate(order)}

def rankings(hits,ll,sim):
    keys=[r['key'] for r in hits]
    terms={'pool':{k:1/(12+j) for j,k in enumerate(keys)}}
    for field in ['upr','paired','whole_gain','recurrent_gain']:
        terms[field]=ranks(keys,{k:ll[k][field] for k in keys})
    for field in ['current','paired']:
        terms[field+'_card']=ranks(keys,{k:sim[k][field] for k in keys})
    features=dict(expanded=['pool'],upr=['upr'],hypothetical=['pool','current_card','upr'],
                  paired=['pool','paired_card','paired'],
                  whole=['pool','paired_card','paired','whole_gain'],
                  recurrent=['pool','paired_card','paired','recurrent_gain'])
    return {m:sorted(keys,key=lambda k:-sum(terms[t][k] for t in parts)) for m,parts in features.items()}

def prompt(question,records,cards,mode):
    entries=[]
    for row in records:
        title=row['title'];txt='Page: '+title+'\nCurrent evidence: '+row['current']
        if mode in ('hypothetical','paired','whole','recurrent'):
            key='current' if mode=='hypothetical' else 'paired'
            c=cards[row['key']][key]
            if key=='paired':
                txt+='\nEarlier evidence: '+row['old']
            if c['valid']:
                txt+='\nRelated question: '+c['question']+'\nSuggested current answer: '+c['current']
        entries.append(txt)
    return ('Use the evidence to answer the user question for the current edition. '
            'Return just the complete short answer, including the requested details. '
            'If a suggested answer addresses a different question, answer from the evidence.\n'
            +'\n\n'.join(entries)+'\nUser question: '+question)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    pool=read(HERE/'pool.json');ll=rows(HERE/'likelihoods.jsonl','id');sim=read(HERE/'card_similarity.json')
    cards=rows(HERE/'cards.jsonl','key')
    queries=read(SOURCE/'questions.json');dest=HERE/'answers.jsonl'
    seen=set(rows(dest,'id')) if dest.exists() else set();start=time.monotonic()
    with dest.open('a') as f:
        for j,q in enumerate(queries):
            uid=q['id']
            if uid in seen:continue
            tick=time.monotonic();pos=engine.positions
            ranked=rankings(pool[uid],ll[uid]['scores'],sim[uid])
            records={r['key']:r for r in pool[uid]};answers={};actual={}
            for mode,order in ranked.items():
                text=prompt(q['question'],[records[k] for k in order[:3]],cards,mode)
                if text not in actual:actual[text]=generate(engine,text,40)
                answer=actual[text];answers[mode+'_open']=answer
                if mode in ('hypothetical','paired','whole','recurrent'):
                    c=cards[order[0]]['current' if mode=='hypothetical' else 'paired']
                    answers[mode+'_copy']=c['current'] if c['valid'] else answer
            f.write(json.dumps(dict(id=uid,ranked=ranked,answers=answers,calls=len(actual),positions=engine.positions-pos,
                                    seconds=time.monotonic()-tick),ensure_ascii=False)+'\n');f.flush()
            mx.clear_cache()
            print(j+1,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('answer phase')
if __name__=='__main__':main()
