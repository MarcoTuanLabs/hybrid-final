"""Factorially exchange three cache components and attribute dated-answer log odds."""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import mlx.core as mx

P=Path(__file__).resolve().parent;R=P.parent
sys.path.insert(0,str(R/'revision_pairs'))
from paired import Engine,cache_for,first_distribution,score_label,shortest

CHANNELS=('attention','convolution','recurrent')


def assign(engine,caches,bits):
    """The attention host owns rotary positions and the continuation offset."""
    att,conv,rec=bits;cache=copy.deepcopy(caches[att])
    for i,layer in enumerate(engine.model.model.layers):
        if layer.is_linear:
            cache[i][0]=caches[conv][i][0]
            cache[i][1]=caches[rec][i][1]
    return cache


def attrib(odds):
    phi={}
    for j,channel in enumerate(CHANNELS):
        part=0.
        for bits in itertools.product([0,1],repeat=3):
            if bits[j]:continue
            m=sum(bits);w=math.factorial(m)*math.factorial(2-m)/6
            alt=list(bits);alt[j]=1
            key=lambda bs:''.join(map(str,bs))
            part+=w*(odds[key(alt)]-odds[key(bits)])
        phi[channel]=part
    return phi


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model');ap.add_argument('tag');args=ap.parse_args()
    read=lambda f:json.loads(f.read_text())
    reg=read(P/'contract.json')
    for f,sha in reg['pins'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==sha,f
    mx.set_cache_limit(64*1024**2);engine=Engine(args.model)
    items=read(R/'asof_revision/ranked.json');keys=read(R/'asof_revision/labels.json')
    dest=P/(args.tag+'_rows.jsonl')
    seen={r['id'] for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    started=time.monotonic()
    with dest.open('a') as out:
        for j,item in enumerate(items):
            uid=item['id']
            if uid in seen:continue
            t=time.monotonic();base=engine.positions
            caches=[];tails=[]
            for rows in item['evidence']:
                text='Use the archived entry to answer the question. Give only the shortest factual answer.\nPage: '+item['page']+'\n'
                text+='\n'.join(r['text'] for r in rows)+'\nQuestion: '+item['question']
                cache,tail=cache_for(engine,text);caches.append(cache);tails.append(tail)
            assert tails[0]==tails[1]
            odds={};scores={};free={};null=[]
            for bs in itertools.product([0,1],repeat=3):
                tag=''.join(map(str,bs));c=assign(engine,caches,bs)
                b=first_distribution(engine,c,tails[0])
                ll=[score_label(engine,b,str(a)) for a in keys[uid]]
                odds[tag]=ll[1]-ll[0];scores[tag]=ll
                if len(set(bs))==1:
                    own=first_distribution(engine,caches[bs[0]],tails[0])
                    null.extend([abs(x-score_label(engine,own,str(a))) for a,x in zip(keys[uid],ll,strict=True)])
                    free[tag]=shortest(engine.answer(tails[0],copy.deepcopy(caches[bs[0]]),stop=40)[0])
                del b,c
            phi=attrib(odds);gap=odds['111']-odds['000']
            assert abs(sum(phi.values())-gap)<1e-6
            assert max(null)<1e-5,max(null)
            out.write(json.dumps(dict(id=uid,log_odds=odds,log_scores=scores,contribution=phi,total_gap=gap,
                       identity_max_error=max(null),open_answers=free,positions=engine.positions-base,shared_wall_s=time.monotonic()-t),ensure_ascii=False)+'\n')
            out.flush();print(j+1,round(time.monotonic()-started,1),flush=True);del caches
            mx.clear_cache()
            if time.monotonic()-started>reg['deadline_s']:raise TimeoutError('registered budget')
    (P/(args.tag+'_receipt.json')).write_text(json.dumps(dict(n=len(items),wall_s=time.monotonic()-started,positions=engine.positions,prior=len(seen)))+'\n')
if __name__=='__main__':main()
