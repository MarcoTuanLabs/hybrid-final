"""Compare chronological replay and cache repair with a dated append-only conversation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

P=Path(__file__).resolve().parent
R=P.parent
spec=importlib.util.spec_from_file_location('retention_replay',R/'retention_splice/run.py')
replay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class BoundedEngine(replay.Engine):
    def body(self,ids,cache,answer=False,retention=None):
        if answer:
            return super().body(ids,cache,answer=True)
        for start in range(0,len(ids),128):
            piece=[] if retention is not None else None
            super().body(ids[start:start+128],cache,retention=piece)
            if piece is not None:
                if retention:
                    retention[:]=[a*b for a,b in zip(retention,piece,strict=True)]
                    mx.eval(*retention)
                else:
                    retention.extend(piece)


def main():
    reg=json.loads((P/'contract.json').read_text())
    for name,sha in reg['pins'].items():
        assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha,name
    mx.set_cache_limit(64*1024**2)
    model,tok=load(reg['model'])
    engine=BoundedEngine(model,tok)
    encode=lambda text:tok.encode(text,add_special_tokens=False)
    template=tok.apply_chat_template([dict(role='user',content=replay.PREFIX+'SPLITPOINT')],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    before,after=template.split('SPLITPOINT')
    opening,closing=encode(before),encode(after)
    cache=make_prompt_cache(model);engine.reset();engine.decode(opening[:10],cache);del cache
    cases=json.loads((R/'suffix_transport/inputs.json').read_text())
    dest=P/'rows.jsonl'
    done=set()
    if dest.exists():
        old=list(map(json.loads,dest.read_text().splitlines()))
        for uid in {r['id'] for r in old}:
            assert {r['mode'] for r in old if r['id']==uid}==set(reg['modes'])
            done.add(uid)
    start=time.monotonic()
    with dest.open('a') as stream:
        for case in cases:
            if case['id'] in done:continue
            order=reg['modes'].copy();random.Random(case['id']).shuffle(order)
            records=[]
            for mode in order:
                engine.reset();timer=time.perf_counter()
                prior=encode('\n'.join(case['prior'])+'\n')
                later=encode('\n'.join(case['later']))[:512]
                prefix=make_prompt_cache(model);engine.body(opening+prior,prefix)
                retention=[]
                if mode=='gates':
                    boundary=copy.deepcopy(prefix);engine.body(later[:16],boundary)
                    old=copy.deepcopy(boundary);engine.body(later[16:],old,retention=retention)
                else:
                    old=copy.deepcopy(prefix) if mode=='full' else prefix
                    engine.body(later,old)
                mx.synchronize();ingest=time.perf_counter()-timer
                update=time.perf_counter()
                insert_text='\n'.join(case['insert'])+'\n'
                q=encode('\n\nQuestion on '+case['date']+': '+case['question'])+closing
                if mode=='full':
                    live=prefix;engine.body(encode(insert_text)+later,live)
                elif mode=='gates':
                    new=copy.deepcopy(prefix);engine.body(encode(insert_text)+later[:16],new)
                    live=replay.splice(model,old,boundary,new,[],'stale')
                    it=iter(retention)
                    for i,layer in enumerate(model.model.layers):
                        if layer.is_linear:
                            live[i][1]=old[i][1]+(new[i][1]-boundary[i][1])*next(it)[:,:,None,None]
                    mx.eval(*[live[i][1] for i,layer in enumerate(model.model.layers) if layer.is_linear])
                else:
                    label='\n' if mode=='append' else '\nLate-arriving conversation record. Use its stated date to place it in the timeline.\n'
                    live=old;engine.body(encode(label+insert_text),live)
                seq,_=engine.decode(q,live);mx.synchronize()
                records.append(dict(id=case['id'],mode=mode,text=tok.decode(seq).strip(),seq=seq,
                                    positions=engine.positions,ingest_s=ingest,update_answer_s=time.perf_counter()-update))
                del live,old,prefix
                if mode=='gates':del new,boundary
                mx.clear_cache()
            for record in records:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            stream.flush();print(case['id'],round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>reg['deadline_s']:raise TimeoutError('pilot budget')
    (P/'receipt.json').write_text(json.dumps(dict(n=len(cases),modes=reg['modes'],elapsed_s=time.monotonic()-start,prior=len(done)))+'\n')
if __name__=='__main__':main()
