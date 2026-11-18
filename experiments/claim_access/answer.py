"""Generate a fresh answer for each fixed evidence selection with the same chronological reader."""
import argparse
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from query import Engine,generate

def blend(orders,keys):
    scores={k:0.0 for k in keys}
    for order in orders:
        for rank,k in enumerate(order):
            scores[k]+=1/(8+rank)
    return sorted(keys,key=lambda k:-scores[k])

def coverage(order,docs):
    reserved=[k for i in range(4) for k in [j for j in order if docs[j]['quarter']==i][:2]]
    return list(dict.fromkeys(reserved+order))[:8]

def selections(row,support):
    docs=row['docs'];keys=row['support'];span=row['span_rank']
    plain=sorted(keys,key=lambda k:-support[k]['active'])
    ranks=[sorted(keys,key=lambda k:-docs[k]['margin']),plain]
    orders=dict(lexdense=row['rank'][:8],span=span[:8],span_quarters=coverage(span,docs),wide=row['rank'],all_scored=keys)
    orders['lm']=coverage(blend(ranks,keys),docs)
    for field in ['whole_gain','matrix_gain']:
        gain=sorted(keys,key=lambda k:-support[k][field])
        orders[field]=coverage(blend(ranks+[gain],keys),docs)
    return orders

def request(q,records):
    records=sorted({(v['session'],v['offset']):v for v in records}.values(),key=lambda d:(d['session'],d['offset']))
    evidence='\n'.join(v['date']+'  User: '+v['text'] for v in records)
    return ("Answer the question from these dated user reports. Identify who made each claim and whom it describes. "
            "For a changing fact, state the prior and new values in order. For a conditional preference, give the specific condition. "
            "For stable biographical facts the benchmark may introduce a later contradictory report. Prefer the earliest directly supported report unless independently resolved, and mention the competing report. "
            "Give a concise, complete answer on one line.\n"+evidence+'\nAs of '+q['date']+'\nQuestion: '+q['question'])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('data',type=Path);a=ap.parse_args()
    qs=json.loads((a.data/'questions.json').read_text());by_uid={q['id']:q for q in qs}
    reads=list(map(json.loads,(HERE/'retrieval.jsonl').read_text().splitlines()));supports={r['id']:r['scores'] for r in map(json.loads,(HERE/'supports.jsonl').read_text().splitlines())}
    assert set(by_uid)==set(supports)=={r['id'] for r in reads}
    engine=Engine(a.model);mx.set_cache_limit(64*1024**2)
    start=time.monotonic();dest=HERE/'answers.jsonl';seen={x['id'] for x in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    with dest.open('a') as out:
        for row in reads:
            uid=row['id']
            if uid in seen:continue
            choices=selections(row,supports[uid]);cached={};answers={};actual={};t=time.monotonic()
            for method,keys in choices.items():
                text=request(by_uid[uid],[row['docs'][k] for k in keys]+row['current'])
                if text not in cached:
                    tick=time.monotonic();n=engine.positions
                    cached[text]=dict(answer=generate(engine,text,80),positions=engine.positions-n,seconds=time.monotonic()-tick)
                actual[method]=cached[text];answers[method]=cached[text]['answer']
            out.write(json.dumps(dict(id=uid,choices=choices,answers=answers,measurements=actual,distinct_prompts=len(cached),seconds=time.monotonic()-t),ensure_ascii=False)+'\n');out.flush()
            print(len(seen)+1,round(time.monotonic()-start,1),flush=True);seen.add(uid);mx.clear_cache()
            if time.monotonic()-start>1200:raise TimeoutError('Answer generation budget')
if __name__=='__main__':main()
