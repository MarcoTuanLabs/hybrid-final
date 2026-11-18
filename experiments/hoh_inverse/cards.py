"""Produce ordinary and cross-edition QA cards blind to the eventual user question."""
import argparse
import json
from pathlib import Path
import time
import mlx.core as mx
from query import Engine,generate

HERE=Path(__file__).resolve().parent

def card(engine,row,paired):
    if paired:
        instruction=('Write one specific factual question with different answers in these two editions of the same page. '
                     'The question must work for both editions without dates unless a date is part of the fact. '
                     'Return exactly one line: question || complete earlier answer || complete updated answer. '
                     'Keep answers brief and preserve units or qualifiers needed to identify the answer.')
        evidence='\nEarlier: '+row['old']+'\nUpdated: '+row['current']
    else:
        instruction=('Write one specific factual question that this passage answers, followed by its complete short answer. '
                     'Return exactly one line: question || answer. Include units or qualifiers needed to identify the answer.')
        evidence='\nPassage: '+row['current']
    raw=generate(engine,instruction+'\nPage: '+row['title']+evidence,56)
    parts=[s.strip() for s in raw.split('||')]
    valid=len(parts)==(3 if paired else 2) and all(parts)
    return dict(raw=raw,valid=bool(valid),question=parts[0] if valid else '',
                old=parts[1] if valid and paired else '',current=parts[-1] if valid else '')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    pool=json.loads((HERE/'pool.json').read_text());sources={r['key']:r for hits in pool.values() for r in hits}
    dest=HERE/'cards.jsonl';seen={r['key'] for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    start=time.monotonic()
    with dest.open('a') as out:
        for j,(key,row) in enumerate(sources.items()):
            if key in seen:continue
            t=time.monotonic();n=engine.positions
            entry=dict(key=key,current=card(engine,row,False),paired=card(engine,row,True))
            out.write(json.dumps(dict(**entry,positions=engine.positions-n,seconds=time.monotonic()-t),ensure_ascii=False)+'\n');out.flush()
            mx.clear_cache()
            if (j+1)%10==0:print(j+1,len(sources),round(time.monotonic()-start,2),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('card phase')
if __name__=='__main__':main()
