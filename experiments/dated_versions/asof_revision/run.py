"""Remove the influence of superseded revision spans along the observed recurrent trajectory."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import make_prompt_cache

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'bridge_reads'))
from engine import Engine

METHODS=('all','attention','values','skip','attention_values','attention_skip','snapshot')
HEADER=('You have dated snapshots of a Wikipedia page. Answer the question using the specified edition. '
        'Return only the shortest factual answer, with no explanation.\n')


def read(path):
    return json.loads(path.read_text())


def region(engine,ids,cache):
    for start in range(0,len(ids),96):
        h=engine.model.model(mx.array([ids[start:start+96]]),cache=cache)
        engine.positions+=min(96,len(ids)-start)
        mx.eval(h)


class RevisionCache:
    def __init__(self,engine,header):
        self.engine=engine
        self.cache=make_prompt_cache(engine.model)
        self.layers=[i for i,l in enumerate(engine.model.model.layers) if l.is_linear]
        self.offset=len(header);self.bounds={}
        region(engine,header,self.cache)
        initial=[self.cache[i][1] for i in self.layers]
        self.states={(target,mode):initial.copy() for target in range(2) for mode in ('values','skip')}
        self.extra_steps=0

    def append(self,tokens,edition):
        self.bounds[edition]=self.offset,self.offset+len(tokens)
        self.offset+=len(tokens)
        kernel=qwen.gated_delta_update
        first=len(self.bounds)==1
        for start in range(0,len(tokens),96):
            part=tokens[start:start+96]
            layer=0
            def capture(q,k,v,a,b,al,bias,state,mask,**kw):
                nonlocal layer
                output,new=kernel(q,k,v,a,b,al,bias,state,mask,**kw)
                for (wanted,mode),states in self.states.items():
                    if wanted==edition and first:
                        states[layer]=new
                    elif wanted==edition or mode=='values':
                        val=v if wanted==edition else mx.zeros_like(v)
                        _,states[layer]=kernel(q,k,val,a,b,al,bias,states[layer],mask,**kw)
                        self.extra_steps+=len(part)
                layer+=1
                return output,new
            qwen.gated_delta_update=capture
            try:
                region(self.engine,part,self.cache)
                mx.eval(*[x for states in self.states.values() for x in states])
            finally:
                qwen.gated_delta_update=kernel
            assert layer==len(self.layers)

    def answer(self,target,mode,tokens):
        cache=copy.deepcopy(self.cache)
        stripped=mode.removeprefix('attention_')
        if stripped in ('values','skip'):
            for layer,state in zip(self.layers,self.states[target,stripped],strict=True):
                cache[layer][1]=state
        normal=qwen.create_attention_mask
        lo,hi=self.bounds[1-target]
        def restrict(hidden,cache,**kwargs):
            seq=hidden.shape[1]
            loc=mx.arange(cache.offset+seq)
            return create_causal_mask(seq,offset=cache.offset) & ((loc<lo)|(loc>=hi))
        if mode.startswith('attention'):
            qwen.create_attention_mask=restrict
        try:
            return self.engine.answer(tokens,cache,stop=40)[0]
        finally:
            qwen.create_attention_mask=normal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model');parser.add_argument('tag');args=parser.parse_args()
    contract=read(P/'contract.json')
    for f,sha in contract['pins'].items():
        assert hashlib.sha256((P/f).read_bytes()).hexdigest()==sha,f
    mx.set_cache_limit(64*1024**2)
    engine=Engine(args.model)
    shell=engine.chat(HEADER+'SPLITPOINT')
    before,after=shell.split('SPLITPOINT')
    prefix=engine.encode(before)
    file=P/(args.tag+'_rows.jsonl');done=set()
    if file.exists():
        done={r['id'] for r in map(json.loads,file.read_text().splitlines())}
    start=time.monotonic()
    with file.open('a') as stream:
        for index,item in enumerate(read(P/'ranked.json')):
            if item['id'] in done:continue
            base=engine.positions;t=time.monotonic()
            chunks=[engine.encode('\nEdition at '+date+' — '+item['page']+'\n'+'\n'.join(row['text'] for row in rows)+'\n')
                    for rows,date in zip(item['evidence'],item['dates'],strict=True)]
            order=(0,1) if index%2==0 else (1,0)
            full=RevisionCache(engine,prefix)
            for ed in order:full.append(chunks[ed],ed)
            preds={}
            for target,date in enumerate(item['dates']):
                question=engine.encode('\nAccording to the '+date+' edition, '+item['question']+after)
                values={}
                for mode in METHODS:
                    if mode=='snapshot':
                        snap=make_prompt_cache(engine.model)
                        region(engine,prefix,snap);region(engine,chunks[target],snap)
                        answer=engine.answer(question,snap,stop=40)[0];del snap
                    else:
                        answer=full.answer(target,mode,question)
                    values[mode]=answer.strip().strip('*`').strip()
                preds[target]=values
            stream.write(json.dumps(dict(id=item['id'],answers=preds,order=order,positions=engine.positions-base,
                        retained_payload_tokens=list(map(len,chunks)),auxiliary_layer_positions=full.extra_steps,
                        shared_wall_s=time.monotonic()-t),ensure_ascii=False)+'\n');stream.flush()
            del full;mx.clear_cache()
            print(index+1,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>contract['deadline_s']:raise TimeoutError('pilot budget')
    (P/(args.tag+'_receipt.json')).write_text(json.dumps(dict(elapsed_s=time.monotonic()-start,positions=engine.positions,previous=len(done)))+'\n')
if __name__=='__main__':main()
