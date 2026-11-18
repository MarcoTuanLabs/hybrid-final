"""Perform independent full lifecycles with one old history and one subsequently revised query."""
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm.models.qwen3_5 as module
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import compute_g

p=Path(__file__).resolve().parent
sys.path.insert(0,str(p.parent/'suffix_transport'))
from run import PREFIX, retrieve, splice

def read(f):return json.loads(f.read_text())

class Engine:
    def __init__(self,model,tokenizer):
        self.model=model
        self.tok=tokenizer
        self.config=model.language_model.args
        self.kernel=module.gated_delta_update

    def reset(self):
        self.positions=0
        self.macs=0

    def body(self,ids,cache,answer=False,retention=None):
        cfg=self.config;n=len(ids)
        full_layers=cfg.num_hidden_layers//cfg.full_attention_interval
        linear_layers=cfg.num_hidden_layers-full_layers
        off=next(c.offset for c in cache if hasattr(c,'keys_and_values'))
        dk=cfg.linear_num_key_heads*cfg.linear_key_head_dim
        dv=cfg.linear_num_value_heads*cfg.linear_value_head_dim
        da=cfg.num_attention_heads*cfg.head_dim
        dkv=cfg.num_key_value_heads*cfg.head_dim
        hidden=cfg.hidden_size
        mlp=3*hidden*cfg.intermediate_size
        projections=hidden*(2*dk+3*dv+2*cfg.linear_num_value_heads)
        convolution=cfg.linear_conv_kernel_dim*(2*dk+dv)
        recurrence=4*dv*cfg.linear_key_head_dim
        # Dense-equivalent multiply-accumulates, including a generous four matrix sweeps per recurrent step.
        self.macs+=n*(cfg.num_hidden_layers*mlp+linear_layers*(projections+convolution+recurrence)+full_layers*hidden*(3*da+2*dkv))
        self.macs+=full_layers*2*da*(n*off+n*(n+1)//2)
        self.positions+=n
        def capture(q,k,v,a,b,al,bias,state,mask,**kw):
            retention.append(mx.prod(compute_g(al,a,bias),axis=1))
            return self.kernel(q,k,v,a,b,al,bias,state,mask,**kw)
        if retention is not None:module.gated_delta_update=capture
        try:
            hs=self.model.model(mx.array([ids]),cache=cache)
            if answer:
                last=hs[:,-1:,:];lm=self.model.language_model
                logits=lm.model.embed_tokens.as_linear(last) if cfg.tie_word_embeddings else lm.lm_head(last)
                self.macs+=hidden*cfg.vocab_size
                mx.eval(logits)
                return logits[0,0]
            mx.eval(hs)
            if retention is not None:mx.eval(*retention)
        finally:module.gated_delta_update=self.kernel

    def decode(self,ids,cache):
        logits=self.body(ids,cache,True)
        first=np.asarray(logits.astype(mx.float32));output=[]
        for _ in range(28):
            token=int(mx.argmax(logits).item())
            if token in self.tok.eos_token_ids:break
            output.append(token)
            if len(output)==28 or '\n' in self.tok.decode(output):break
            logits=self.body([token],cache,True)
        mx.synchronize()
        return output,first

def main():
    reg=read(p/'contract.json')
    for name,sha in reg['pins'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
    model_dir=Path(reg['model']);manifest=read(model_dir/'receipt.json')
    assert manifest['revision']==reg['revision']
    for name,info in manifest['files'].items():
        with (model_dir/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==info['sha256'],name
    model,tok=load(str(model_dir));eng=Engine(model,tok)
    enc=lambda x:tok.encode(x,add_special_tokens=False)
    template=tok.apply_chat_template([dict(role='user',content=PREFIX+'SPLITPOINT')],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    before,after=template.split('SPLITPOINT');open_ids=enc(before);close_ids=enc(after)
    # Warm kernels outside all method timings.
    c=make_prompt_cache(model);eng.reset();eng.decode(open_ids[:10],c);del c
    file=p/'rows.jsonl';done=set()
    if file.exists():
        seen=list(map(json.loads,file.read_text().splitlines()))
        for uid in {x['id'] for x in seen}:
            assert {x['mode'] for x in seen if x['id']==uid}==set(reg['modes'])
            done.add(uid)
    examples=read(p/'inputs.json');start=time.perf_counter()
    with file.open('a') as out:
        for row in examples:
            if row['id'] in done:continue
            records=[];preds={};modes=reg['modes'].copy();random.Random(row['id']).shuffle(modes)
            for mode in modes:
                eng.reset();t=time.perf_counter()
                # Tokenization, CPU ranking, prompt assembly, device synchronization and cache manipulation are timed.
                prior=enc('\n'.join(row['prior'])+'\n')
                later=enc('\n'.join(row['later']))[:512];h=16
                prefix=make_prompt_cache(model);g=[]
                if mode!='bm25':
                    eng.body(open_ids+prior,prefix)
                    if mode=='full':
                        old=copy.deepcopy(prefix);eng.body(later,old)
                    else:
                        bound=copy.deepcopy(prefix);eng.body(later[:h],bound)
                        old=copy.deepcopy(bound);eng.body(later[h:],old,retention=g if mode=='gates' else None)
                mx.synchronize();ingest_s=time.perf_counter()-t;ingest_macs=eng.macs;ingest_positions=eng.positions
                online=time.perf_counter();insert=enc('\n'.join(row['insert'])+'\n')
                qids=enc('\n\nQuestion on '+row['date']+': '+row['question'])+close_ids
                if mode=='bm25':
                    blocks=row['prior']+row['insert']+tok.decode(later).splitlines()
                    qids=open_ids+enc('\n'.join(retrieve(blocks,row['question'])))+qids
                    live=prefix
                elif mode=='full':
                    live=copy.deepcopy(prefix);eng.body(insert+later,live)
                else:
                    updated=copy.deepcopy(prefix);eng.body(insert+later[:h],updated)
                    live=splice(model,old,bound,updated,[], 'stale')
                    if mode=='gates':
                        j=0
                        for i,layer in enumerate(model.model.layers):
                            if layer.is_linear:
                                live[i][1]=old[i][1]+(updated[i][1]-bound[i][1])*g[j][:,:,None,None];j+=1
                        mx.eval(*[c[1] for i,c in enumerate(live) if model.model.layers[i].is_linear]);assert j==len(g)
                seq,logits=eng.decode(qids,live);online_s=time.perf_counter()-online
                # Lower order additions, decays, RoPE key shifts and activation functions are timed. Record scalar-repair work separately.
                heads=sum(v.size for v in g)
                repair_scalars=heads*(len(later)-h)*12+heads*eng.config.linear_key_head_dim*eng.config.linear_value_head_dim*3
                preds[mode]=logits
                records.append(dict(id=row['id'],mode=mode,seq=seq,text=tok.decode(seq).strip(),ingest_s=ingest_s,online_s=online_s,
                   ingest_macs=ingest_macs,online_macs=eng.macs-ingest_macs,ingest_positions=ingest_positions,online_positions=eng.positions-ingest_positions,
                   saved_retention_bytes=sum(v.nbytes for v in g),estimated_repair_scalar_ops=repair_scalars))
                del live,prefix
                if mode!='bm25':del old
                if mode in ('stale','gates'):del updated,bound
            for item in records:
                diff=preds[item['mode']]-preds['full'];item['rmse']=float(np.sqrt(np.mean((diff-diff.mean())**2)))
                out.write(json.dumps(item,ensure_ascii=False)+'\n')
            out.flush();print(row['id'],round(time.perf_counter()-start,1),flush=True);mx.clear_cache()
    (p/'receipt.json').write_text(json.dumps(dict(finished=True,cases=len(examples),metrics='independent old-history ingest and update, one query after each edit',wall_s=time.perf_counter()-start))+'\n')
if __name__=='__main__':main()
