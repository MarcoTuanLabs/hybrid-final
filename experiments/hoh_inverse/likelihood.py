"""Score the user's question as a continuation of genuine and counterfactual revision memories."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
from query import Engine,history,SOURCE

HERE=Path(__file__).resolve().parent

def likelihood(engine,cache,tail,question):
    target=engine.encode(question)
    assert target
    ids=tail+target[:-1]
    states=engine.model.model(mx.array([ids]),cache=cache)
    engine.positions+=len(ids)
    states=states[:,len(tail)-1:]
    lm=engine.language
    logits=(lm.model.embed_tokens.as_linear(states) if lm.args.tie_word_embeddings else lm.lm_head(states))[0].astype(mx.float32)
    ll=mx.take_along_axis(logits,mx.array(target)[:,None],axis=1)[:,0]-mx.logsumexp(logits,axis=1)
    return float(mx.mean(ll).item())

def score(engine,row,question):
    direct='Write a specific factual question that is answered by the following Wikipedia entry. Output the question only.\nPage: '+row['title']+'\nEntry: '+row['current']
    head=('Write a factual question about the same property in both Wikipedia editions. Output only the question.\n'
          +'Page: '+row['title']+'\nEarlier: ')
    actual=head+row['old']+'\nLater: '+row['current']
    null=head+row['current']+'\nLater: '+row['current']
    c,t=history(engine,direct)
    upr=likelihood(engine,c,t,question);del c
    ca,tail=history(engine,actual);cn,blank=history(engine,null)
    assert tail==blank
    mixed=copy.deepcopy(ca)
    for j,layer in enumerate(engine.model.model.layers):
        if layer.is_linear:
            mixed[j][1]=cn[j][1]
    native=likelihood(engine,mixed,tail,question)
    both=likelihood(engine,ca,tail,question)
    zero=likelihood(engine,cn,tail,question)
    return dict(upr=upr,paired=both,null=zero,recurrent_null=native,whole_gain=both-zero,recurrent_gain=both-native)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    qs=json.loads((SOURCE/'questions.json').read_text());pool=json.loads((HERE/'pool.json').read_text())
    dest=HERE/'likelihoods.jsonl';seen={r['id'] for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    start=time.monotonic()
    with dest.open('a') as f:
        for j,q in enumerate(qs):
            if q['id'] in seen:continue
            t=time.monotonic();n=engine.positions;items={}
            for row in pool[q['id']]:
                items[row['key']]=score(engine,row,q['question']);mx.clear_cache()
            f.write(json.dumps(dict(id=q['id'],scores=items,positions=engine.positions-n,seconds=time.monotonic()-t))+'\n');f.flush()
            print(j+1,round(time.monotonic()-start,2),flush=True)
            if time.monotonic()-start>1800:raise TimeoutError('question likelihood phase')
if __name__=='__main__':main()
