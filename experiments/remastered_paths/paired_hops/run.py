"""Propagate the value before an edit alongside the new value during multihop answering."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm.models.qwen3_5 as native

P = Path(__file__).resolve().parent
R = P.parent
sys.path.insert(0, str(R / 'bridge_reads'))
from bridge import Bank, boundary, canonical
from engine import Engine


def read(path):
    return json.loads(path.read_text())


def strip_answer(text):
    return text.strip('*` ').rstrip('.')


def first_queries(engine, prefix, tail):
    captured = []
    kernel = native.gated_delta_update
    def record(q,k,v,*args,**kwargs):
        rep = v.shape[2] // q.shape[2]
        captured.append(mx.repeat(q[:,0].astype(mx.float32),rep,axis=1))
        return kernel(q,k,v,*args,**kwargs)
    native.gated_delta_update = record
    try:
        logits = engine.forward(tail[:1],copy.deepcopy(prefix))
        mx.eval(logits,*captured)
    finally:
        native.gated_delta_update = kernel
    return iter(captured)


def adjust(engine, current, original, queries, mode):
    cache = copy.deepcopy(current)
    for i, layer in enumerate(engine.model.model.layers):
        if not layer.is_linear:
            continue
        state = current[i][1]
        prior = original[i][1]
        q = next(queries)
        q = q * mx.rsqrt(mx.sum(q*q,axis=-1,keepdims=True)+1e-10)
        new = (state @ q[...,None])[...,0].astype(mx.float32)
        old = (prior @ q[...,None])[...,0].astype(mx.float32)
        change = new - old
        if mode == 'tangent':
            scale = mx.sum(new*change,axis=-1,keepdims=True)/(mx.sum(new*new,axis=-1,keepdims=True)+1e-10)
            change = change - scale*new
        bound = mx.sqrt(mx.sum(new*new,axis=-1,keepdims=True)/(mx.sum(change*change,axis=-1,keepdims=True)+1e-10))
        increment = (change*mx.minimum(1.,bound))[...,None]*q[...,None,:]
        cache[i][1] = (state+increment).astype(state.dtype)
    return cache


def contrast(engine, prefix, old, tail):
    caches = [copy.deepcopy(prefix),copy.deepcopy(old)]
    out = []
    for _ in range(24):
        now,before = [engine.forward(tail,c) for c in caches]
        delta = now+(now-before)
        logits = mx.where(now>=mx.max(now)-2.302585093,delta,-float('inf'))
        token = int(mx.argmax(logits).item())
        if token in engine.tokenizer.eos_token_ids:
            break
        out.append(token)
        if '\n' in engine.tokenizer.decode(out):
            break
        tail = [token]
    return strip_answer(engine.tokenizer.decode(out).split('\n',1)[0])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model'); ap.add_argument('tag')
    args=ap.parse_args()
    reg=read(P/'contract.json')
    for name,sha in reg['pins'].items():
        assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha,name
    methods=reg['methods']
    engine=Engine(args.model)
    schema=read(R/'typed_reads/schema.json')
    items={x['id']:x for x in read(R/'state_parser/items.json') if x['split']=='exposed'}
    plans=read(R/'clause_reads/4b_parser_scores.json')['stats']['lexical']['exposed_id_plans']
    fname=P/(args.tag+'_rows.jsonl')
    done={r['id'] for r in map(json.loads,fname.read_text().splitlines())} if fname.exists() else set()
    started=time.monotonic()
    transfer={r['id']:r for r in map(json.loads,(R/'walk_transfer'/(args.tag+'_rows.jsonl')).read_text().splitlines())}
    assert len(transfer)==len(items)
    with fname.open('a') as f:
        for bi,block in enumerate(read(R/'remastered/banks.json')):
            bank=Bank(block['documents'])
            for j,row in enumerate(block['queries']):
                uid=row['id']
                if uid in done:
                    continue
                source=transfer[uid]
                trace={}; outcomes={}; calls={}; positions=engine.positions
                def factual(entity,rel):
                    q=schema[rel].format(entity)
                    if q not in calls:
                        calls[q]=engine.factual(q)
                    return calls[q]
                def changed(new,old,rel):
                    key=json.dumps([new,old,rel],ensure_ascii=False)
                    if key in trace:
                        return trace[key]
                    pre,tail=boundary(engine,schema[rel].format(new))
                    other,same=boundary(engine,schema[rel].format(old))
                    assert tail==same
                    c=engine.prefill(pre); o=engine.prefill(other)
                    queries=list(first_queries(engine,c,tail))
                    values={'plain':factual(new,rel),'whole':contrast(engine,c,o,tail)}
                    for mode in ('projected','tangent'):
                        repaired=adjust(engine,c,o,iter(queries),mode)
                        values[mode]=strip_answer(engine.answer(tail,repaired)[0])
                    trace[key]=values
                    return values
                for mode in methods:
                    if source['route']!='context':
                        outcomes[mode]=source['reference']
                        continue
                    cur=prior=items[uid]['root']
                    for rel in plans[uid]:
                        ix=bank.link(cur,rel)
                        jx=bank.link(prior,rel)
                        prev=prior
                        prior=bank.documents[jx]['old'] if jx is not None else factual(prior,rel)
                        if ix is not None:
                            cur=bank.documents[ix]['obj']
                        elif canonical(cur)==canonical(prev) or mode=='plain':
                            cur=factual(cur,rel)
                        else:
                            cur=changed(cur,prev,rel)[mode]
                    outcomes[mode]=cur
                rec=dict(id=uid,bank=bi,outputs=outcomes,contrasts=trace,shared_positions=engine.positions-positions,
                         plain_replication=(outcomes['plain']==source['outputs']['lexical']['answer']))
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                if j%12==11:
                    print(bi,j+1,round(time.monotonic()-started,1),flush=True)
                if time.monotonic()-started>reg['deadline_s']:
                    raise TimeoutError('batch deadline')
    (P/(args.tag+'_receipt.json')).write_text(json.dumps(dict(shared_positions=engine.positions,wall_s=time.monotonic()-started,prior_rows=len(done)),indent=2)+'\n')


if __name__=='__main__':
    main()
