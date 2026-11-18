"""Measure each complete update before answering two future questions from the updated cache."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from documents import HERE,MARK,setup,message,document,template,encoding,future
from cache_service import advance,copy_at,elapsed,same,size
from wire import apply_patch,patch,prefix,encode_state,decode_state
from stream import transfer

def pack_bits(q,width):
    stride=8//width
    flat=q.reshape(*q.shape[:-1],-1,stride).astype(mx.uint32)
    powers=mx.array([1<<(width*i) for i in range(stride)],mx.uint32)
    return mx.sum(flat*powers,axis=-1).astype(mx.uint8)
def unpack(bits,width):
    parts=[(bits>>(width*i))&((1<<width)-1) for i in range(8//width)]
    return mx.stack(parts,axis=-1).reshape(*bits.shape[:-1],-1).astype(mx.float32)
def encode_document(trial,old,new,common,mode,bases):
    meta=[];arrays=[];j=0
    for i,layer in enumerate(trial.model.model.layers):
        name=str(i)
        if not layer.is_linear:
            n=new[i].offset;s=min(common,n,old[i].offset)
            if s and not all(bool(mx.array_equal(a[:,:,:s],b[:,:,:s]).item()) for a,b in [(old[i].keys,new[i].keys),(old[i].values,new[i].values)]):s=0
            meta.append([i,n,s])
            arrays.extend([('k'+name,new[i].keys[:,:,s:n]),('v'+name,new[i].values[:,:,s:n])])
            continue
        arrays.append(('c'+name,new[i][0]))
        if mode=='dense':arrays.append(('m'+name,new[i][1]))
        elif mode!='zero':
            d=(new[i][1]-old[i][1])[0]
            if mode in bases:
                arrays.append(('a'+name,(d@bases[mode][j]).astype(mx.float16)))
            elif mode=='bit2':
                lo=mx.min(d,axis=-1,keepdims=True).astype(mx.float16)
                step=((mx.max(d,axis=-1,keepdims=True)-lo.astype(mx.float32))/3).astype(mx.float16)
                q=mx.clip(mx.round((d-lo.astype(mx.float32))/mx.where(step>0,step.astype(mx.float32),1.0)),0,3).astype(mx.uint8)
                arrays.extend([('l'+name,lo),('s'+name,step),('p'+name,pack_bits(q,2))])
            elif mode=='bit1':
                high=d>=mx.mean(d,axis=-1,keepdims=True)
                def centroid(mask):
                    return (mx.sum(mx.where(mask,d,0.0),axis=-1,keepdims=True)/mx.maximum(1,mx.sum(mask,axis=-1,keepdims=True))).astype(mx.float16)
                arrays.extend([('l'+name,centroid(~high)),('s'+name,centroid(high)),('p'+name,pack_bits(high,1))])
            else:raise ValueError(mode)
        j+=1
    # Materialize the encoded matrices as one device request before serialization.
    mx.eval(*[v for _,v in arrays])
    return encode_state(meta,arrays)
def restore(trial,old,packet,mode,bases):
    meta,arr=decode_state(packet);c=copy.deepcopy(old)
    for i,n,s in meta:
        for prop,key in [('keys','k'),('values','v')]:
            old_array=getattr(old[i],prop);suffix=arr[key+str(i)]
            setattr(c[i],prop,mx.concatenate([old_array[:,:,:s],suffix],axis=2) if s else suffix)
        c[i].offset=n
    j=0
    for i,layer in enumerate(trial.model.model.layers):
        if not layer.is_linear:continue
        name=str(i);c[i][0]=arr['c'+name]
        if mode=='dense':c[i][1]=arr['m'+name]
        elif mode!='zero':
            if mode in bases:d=arr['a'+name].astype(mx.float32)@bases[mode][j].swapaxes(-1,-2)
            else:
                z=unpack(arr['p'+name],2 if mode=='bit2' else 1)
                lo=arr['l'+name].astype(mx.float32);hi=arr['s'+name].astype(mx.float32)
                d=lo+hi*z if mode=='bit2' else mx.where(z>0,hi,lo)
            c[i][1]=old[i][1]+d[None]
        j+=1
    mx.eval(*[l.state for l in c]);return c
def respond(trial,c,question,limit):
    tail=encoding(trial,MARK+template(trial,'Shared document.',question).split(MARK,1)[1])
    clone=copy.deepcopy(c);tokens=[];input_ids=tail;llm=trial.model.language_model
    for _ in range(limit):
        hidden=trial.model.model(mx.array([input_ids]),cache=clone)[:,-1:,:]
        logits=llm.model.embed_tokens.as_linear(hidden) if llm.args.tie_word_embeddings else llm.lm_head(hidden)
        t=int(mx.argmax(logits[0,-1]).item())
        if t in trial.tok.eos_token_ids:break
        tokens.append(t)
        if '\n' in trial.tok.decode(tokens):break
        input_ids=[t]
    return tokens
def evaluate(trial,case,ordering,bases,received):
    start=elapsed();src=case['source'];ot=message(src,case['old']);nt=message(src,case['new'])
    a,b=ot.encode(),nt.encode();old_ids=document(trial,ot);c=make_prompt_cache(trial.model);saved={0:copy.deepcopy(c)}
    for i in range(0,len(old_ids),32):
        group=old_ids[i:i+32];advance(trial.model,c,group);saved[i+len(group)]=copy.deepcopy(c)
    old=c;old_s=elapsed()-start;rr=[]
    for mode in ordering:
        t=elapsed();data={}
        if mode.startswith('patch'):
            blob=patch(a,b);packed=elapsed();packet,nbytes=transfer(blob);net=elapsed()
            new=apply_patch(a,packet);assert new==b
            ids=document(trial,new.decode());same_prefix=prefix(old_ids,ids)
            client,cut=copy_at(saved,same_prefix) if mode=='patch_prefix' else (make_prompt_cache(trial.model),0)
            advance(trial.model,client,ids[cut:]);ready=elapsed()
            donor_t=0;sender_t=packed-t
        else:
            ids=document(trial,nt);same_prefix=prefix(old_ids,ids)
            donor,cut=copy_at(saved,same_prefix);advance(trial.model,donor,ids[cut:]);made=elapsed()
            blob,data=encode_document(trial,old,donor,same_prefix,mode,bases);packed=elapsed()
            packet,nbytes=transfer(blob);net=elapsed()
            client=restore(trial,old,packet,mode,received);ready=elapsed()
            donor_t=made-t;sender_t=packed-made
        # Neither question was passed to encode_document, restore, or document.
        outputs=[respond(trial,client,case[k],12 if src=='babi' else 24) for k in ['q','q_control']]
        end=elapsed()
        rr.append(dict(mode=mode,wire_bytes=nbytes,arrays=data,cut=cut,ready_cold_s=ready-t,ready_warm_s=ready-t-donor_t,cold_s=end-t,warm_s=end-t-donor_t,donor_s=donor_t,sender_s=sender_t,net_s=net-packed,receiver_s=ready-net,answers_s=end-ready,tokens=outputs,texts=[trial.tok.decode(tokens) for tokens in outputs]))
        if mode=='dense':same(trial,client,donor)
        mx.clear_cache()
    for k in ['q','q_control']:
        assert future(trial,nt,case[k],ids)==encoding(trial,MARK+template(trial,'Shared document.',case[k]).split(MARK,1)[1])
    return dict(id=case['id'],source=src,old_tokens=len(old_ids),new_tokens=len(ids),old_setup_seconds=old_s,old_logical_bytes=sum(size(trial,v) for v in saved.values()),arms=rr)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('public',type=Path);a=ap.parse_args()
    trial,sets=setup(a.public,a.model);rng=random.Random(27221);cases=sets['eval'];rng.shuffle(cases)
    fit=json.loads((HERE/'fit.json').read_text());assert not (set(map(tuple,fit['ids']))&{(c['source'],c['id']) for c in cases})
    banks=np.load(HERE/'bases.npz');bases={};received={};install={}
    for m in ['read','random','svd']:
        t=elapsed();original=mx.array(banks[m]);payload,_=encode_state({},[('basis',original)])
        packet,nbytes=transfer(payload);_,parts=decode_state(packet);received[m]=parts['basis'].astype(mx.float32);bases[m]=original.astype(mx.float32)
        mx.eval(received[m],bases[m]);install[m]=dict(seconds=elapsed()-t,wire_bytes=nbytes)
    (HERE/'install.json').write_text(json.dumps(install,indent=2)+'\n')
    warm_ids=document(trial,'One shared warmup document.');cw=advance(trial.model,make_prompt_cache(trial.model),warm_ids);respond(trial,cw,'What is a document?',3)
    path=HERE/'rows.jsonl';seen={ (r['source'],r['id']) for r in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    begin=elapsed()
    with path.open('a') as f:
        for case in cases:
            methods=['patch_full','patch_prefix','dense','zero','bit1','bit2','random','svd','read'];rng.shuffle(methods)
            if (case['source'],case['id']) in seen:continue
            r=evaluate(trial,case,methods,bases,received)
            f.write(json.dumps(r)+'\n');f.flush();print(case['source'],case['id'],round(elapsed()-begin,1),flush=True)
            if elapsed()-begin>1500:raise TimeoutError('Complete service budget')
if __name__=='__main__':main()
