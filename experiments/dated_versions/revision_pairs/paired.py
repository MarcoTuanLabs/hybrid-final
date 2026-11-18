"""Propose and score answer pairs for two real article editions with a channel-specific contrast."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

P=Path(__file__).resolve().parent
R=P.parent
sys.path.insert(0,str(R/'bridge_reads'))
from engine import Engine
from bridge import first_distribution
sys.path.insert(0,str(R/'paired_hops'))
from run import first_queries


def read(f):
    return json.loads(f.read_text())


def shortest(s):
    return s.strip().strip('*`" ').rstrip('.').strip()


def score_label(engine,base,label):
    cache,first=base;toks=engine.encode(label)
    if not toks:return -100.
    eos=engine.encode('<|im_end|>')[0]
    hidden=engine.model.model(mx.array([toks]),cache=copy.deepcopy(cache))
    engine.positions+=len(toks)
    lm=engine.language
    values=(lm.model.embed_tokens.as_linear(hidden) if lm.args.tie_word_embeddings else lm.lm_head(hidden))[0].astype(mx.float32)
    tgt=mx.array([*toks[1:],eos])
    logs=mx.take_along_axis(values,tgt[:,None],axis=1)[:,0]-mx.logsumexp(values,axis=1)
    return float(((first[toks[0]]+mx.sum(logs))/(len(toks)+1)).item())


def cache_for(engine,text):
    ids=engine.encode(engine.chat(text));tok=engine.encode('<think>')
    assert len(tok)==1 and ids.count(tok[0])==1
    split=ids.index(tok[0]);cache=make_prompt_cache(engine.model)
    for start in range(0,split,96):
        h=engine.model.model(mx.array([ids[start:min(start+96,split)]]),cache=cache)
        engine.positions+=min(start+96,split)-start
        mx.eval(h)
    return cache,ids[split:]


def counters(engine,current,other,queries):
    raw=copy.deepcopy(current);tangent=copy.deepcopy(current)
    qi=iter(queries)
    for i,layer in enumerate(engine.model.model.layers):
        if not layer.is_linear:continue
        new=current[i][1];old=other[i][1];q=next(qi)
        q=q*mx.rsqrt(mx.sum(q*q,axis=-1,keepdims=True)+1e-10)
        y=(new@q[...,None])[...,0];d=y-(old@q[...,None])[...,0]
        orth=d-y*mx.sum(y*d,axis=-1,keepdims=True)/(mx.sum(y*y,axis=-1,keepdims=True)+1e-10)
        alpha=mx.minimum(1.,mx.sqrt(mx.sum(y*y,axis=-1,keepdims=True)/(mx.sum(orth*orth,axis=-1,keepdims=True)+1e-10)))
        tangent[i][1]=(new-(alpha*orth)[...,None]*q[...,None,:]).astype(new.dtype)
        raw[i][1]=old
    return raw,tangent


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model');ap.add_argument('tag')
    a=ap.parse_args();reg=read(P/'contract.json')
    for f,v in reg['pins'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==v,f
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    source=R/'asof_revision';items=read(source/'ranked.json');align={r['id']:r['pairs'] for r in read(P/'alignment.json')}
    prev={r['id']:r for r in map(json.loads,(source/(a.tag+'_rows.jsonl')).read_text().splitlines())}
    spans=read(source/'span_answers.json')
    path=P/(a.tag+'_rows.jsonl')
    seen={r['id'] for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    start=time.monotonic()
    with path.open('a') as out:
        for ix,item in enumerate(items):
            uid=item['id']
            if uid in seen:continue
            begin=engine.positions;tick=time.monotonic();edits=align[uid];dates=item['dates'];q=item['question']
            pairs={}
            pairs['all']=tuple(shortest(prev[uid]['answers'][str(i)]['all']) for i in range(2))
            pairs['snapshot']=tuple(shortest(prev[uid]['answers'][str(i)]['snapshot']) for i in range(2))
            pairs['span']=tuple(map(shortest,spans[uid]))
            field=[(shortest(d['old']),shortest(d['new'])) for d in edits if d['kind']=='field' and all(0<len(engine.encode(d[k]))<=20 for k in ('old','new'))][:3]
            change='\n'.join((d['key'] or 'Article')+' | '+d['old']+' -> '+d['new'] for d in edits)
            notes=['\n'.join(r['text'] for r in side[:2]) for side in item['evidence']]
            paired_text=('Answer the same question for two dated editions. The left value precedes the right value. '
                 'Give one line with two short answers separated by a double vertical bar.\nSource page: '+item['page']+
                 '\n'+dates[0]+' -> '+dates[1]+'\n'+change+'\nEarlier excerpt: '+notes[0]+'\nLater excerpt: '+notes[1]+'\nQuestion: '+q)
            c,tail=cache_for(engine,paired_text);line=engine.answer(tail,c,stop=64)[0];del c
            arr=[shortest(x) for x in line.split('||')]
            pairs['joint_prompt']=tuple(arr) if len(arr)==2 and all(arr) else pairs['all']
            hosts=[];suffix=[]
            for side in range(2):
                sidekey='old' if side==0 else 'new'
                facts='\n'.join((d['key'] or 'Article')+' | '+d[sidekey] for d in edits)
                text='Give only the shortest factual answer.\nEdition '+dates[side]+' of '+item['page']+'\n'+facts+'\n'+notes[side]+'\nQuestion: '+q
                cache,t=cache_for(engine,text);hosts.append(cache);suffix.append(t)
            assert suffix[0]==suffix[1]
            pairs['aligned_snapshot']=tuple(shortest(engine.answer(suffix[i],copy.deepcopy(hosts[i]),stop=40)[0]) for i in (0,1))
            pairs['aligned_field']=field[0] if field else pairs['aligned_snapshot']
            options=list(dict.fromkeys([*pairs.values(),*field]))
            eligible=[o for o in options if all(0<len(engine.encode(l))<=32 for l in o)]
            if not eligible:eligible=[pairs['all']]
            names=list(dict.fromkeys(l for o in eligible for l in o))
            ll=[]
            for i in (0,1):
                queries=list(first_queries(engine,hosts[i],suffix[i]))
                raw,tangent=counters(engine,hosts[i],hosts[1-i],queries)
                per={}
                for method,cache in [('own',hosts[i]),('raw',raw),('tangent',tangent)]:
                    dist=first_distribution(engine,cache,suffix[i])
                    per[method]={label:score_label(engine,dist,label) for label in names}
                ll.append(per);del raw,tangent,dist
            score={}
            for mode in ('vote','plain','whole','raw','tangent'):
                values=[]
                for pair in eligible:
                    votes=sum(pair==p for p in pairs.values())
                    positive=sum(ll[i]['own'][pair[i]] for i in (0,1))*.5
                    additive=0.
                    if mode in ('whole','raw','tangent'):
                        for i in (0,1):
                            positive_margin=ll[i]['own'][pair[i]]-ll[i]['own'][pair[1-i]]
                            counter=ll[1-i]['own'] if mode=='whole' else ll[i][mode]
                            negative_margin=counter[pair[i]]-counter[pair[1-i]]
                            additive+=.25*(positive_margin-negative_margin)
                    value=votes if mode=='vote' else positive+additive+.35*math.log1p(votes)
                    values.append(value)
                score[mode]=eligible[int(np.argmax(values))]
            rec=dict(id=uid,raw_joint=line,proposals=pairs,options=eligible,winners=score,positions=engine.positions-begin,shared_s=time.monotonic()-tick)
            out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();del hosts,ll
            mx.clear_cache();print(ix+1,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>reg['deadline_s']:raise TimeoutError('pilot time budget')
    (P/(a.tag+'_receipt.json')).write_text(json.dumps(dict(n=len(items),elapsed_s=time.monotonic()-start,positions=engine.positions,prior=len(seen)))+'\n')
if __name__=='__main__':main()
