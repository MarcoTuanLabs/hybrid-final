"""Measure a complete cache handoff and the competing exact text-patch service."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
from mlx_lm.models.cache import make_prompt_cache
from wire import apply_patch,decode_state,encode_state,patch,prefix,send_local

HERE=Path(__file__).resolve().parent

def elapsed():mx.synchronize();return time.perf_counter()
def advance(model,cache,tokens):
    if tokens:
        hidden=model.model(mx.array([tokens]),cache=cache);mx.eval(hidden,*[layer.state for layer in cache])
    return cache
def copy_at(saved,n):
    cut=max(k for k in saved if k<=n);return copy.deepcopy(saved[cut]),cut
def answer(model,c,tail,limit):
    # Serial suffix matches the existing rank-four read-projection reference.
    tokens=[];s=copy.deepcopy(c)
    for token in tail:
        logits=model.model(mx.array([[token]]),cache=s)[0,-1];mx.eval(logits)
    for _ in range(limit):
        t=int(mx.argmax(logits).item())
        if t in model.tok.eos_token_ids:break
        tokens.append(t)
        if '\n' in model.tok.decode(tokens):break
        logits=model.model(mx.array([[t]]),cache=s)[0,-1]
    return tokens
def capture(trial,new,tail):
    c=copy.deepcopy(new);trace=[];kernel=trial.kernel
    for t in tail:
        slots=[]
        def observe(q,k,v,a,b,al,bias,state,mask,**kw):
            slots.append((q[0,0],k[0,0],mx.sigmoid(b)[0,0]))
            return kernel(q,k,v,a,b,al,bias,state,mask,**kw)
        qwen.gated_delta_update=observe
        try:
            hidden=trial.model.model(mx.array([[t]]),cache=c);mx.eval(hidden)
            trace.append([tuple(np.asarray(x.astype(mx.float32)) for x in slot) for slot in slots])
        finally:qwen.gated_delta_update=kernel
    return trace
def quantize(d):
    low=d.min(axis=2,keepdims=True).astype(np.float16)
    scale=((d.max(axis=2,keepdims=True)-low.astype(np.float32))/3).astype(np.float16)
    code=np.rint((d-low.astype(np.float32))/np.where(scale>0,scale.astype(np.float32),1)).clip(0,3).astype(np.uint8)
    packed=code[:,:,0::4]|(code[:,:,1::4]<<2)|(code[:,:,2::4]<<4)|(code[:,:,3::4]<<6)
    return low,scale,packed
def encode(trial,old,new,tail,common,method,observability):
    meta=[];parts=[];rec=0
    trace=capture(trial,new,tail) if method=='projection' else None
    for i,layer in enumerate(trial.model.model.layers):
        if not layer.is_linear:
            o=int(new[i].offset);share=min(common,int(old[i].offset),o)
            if share and not all(bool(mx.array_equal(a[:,:,:share],b[:,:,:share]).item()) for a,b in [(new[i].keys,old[i].keys),(new[i].values,old[i].values)]):share=0
            meta.append([i,o,share]);parts.extend([('k'+str(i),new[i].keys[:,:,share:o,:]),('v'+str(i),new[i].values[:,:,share:o,:])])
            continue
        parts.append(('c'+str(i),new[i][0]))
        if method=='dense':parts.append(('m'+str(i),new[i][1]))
        else:
            delta=np.asarray((new[i][1]-old[i][1])[0].astype(mx.float32))
            if method=='projection':
                _,w=observability([t[rec] for t in trace],len(delta))
                basis=np.linalg.qr(w,mode='reduced')[0].astype(np.float32)
                parts.extend([('b'+str(i),basis),('f'+str(i),delta@basis)])
            else:
                lo,scale,bits=quantize(delta);parts.extend([('l'+str(i),lo),('s'+str(i),scale),('p'+str(i),bits)])
        rec+=1
    return encode_state(meta,parts)
def recover(trial,old,packet,method):
    meta,arr=decode_state(packet);c=copy.deepcopy(old)
    for i,o,share in meta:
        def joined(prefix_array,suffix):
            return mx.concatenate([prefix_array[:,:,:share,:],suffix],axis=2) if share else suffix
        c[i].keys=joined(old[i].keys,arr['k'+str(i)]);c[i].values=joined(old[i].values,arr['v'+str(i)]);c[i].offset=o
    for i,layer in enumerate(trial.model.model.layers):
        if not layer.is_linear:continue
        n=str(i);c[i][0]=arr['c'+n]
        if method=='dense':c[i][1]=arr['m'+n]
        elif method=='projection':
            delta=arr['f'+n]@arr['b'+n].swapaxes(-1,-2);c[i][1]=old[i][1]+delta[None]
        else:
            bits=arr['p'+n];shifts=[0,2,4,6]
            codes=mx.stack([(bits>>shift)&3 for shift in shifts],axis=-1).reshape(old[i][1].shape[1:])
            c[i][1]=old[i][1]+(arr['l'+n].astype(mx.float32)+arr['s'+n].astype(mx.float32)*codes.astype(mx.float32))[None]
    mx.eval(*[l.state for l in c]);return c

def body(case,key,facts,config):
    if case['source']=='babi':
        return 'Read the story in order. Answer the question with the location only.\n\n'+'\n'.join(facts)+'\n\n'+case[key]
    return config['mquake_instruction']+'\n\nFacts:\n'+'\n'.join(facts)+'\n\nQuestion: '+case[key]+'\nAnswer:'
def tokenize(trial,text):
    fmt=trial.tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    tokens=trial.tok.encode(fmt,add_special_tokens=False);marker,=trial.tok.encode('<think>',add_special_tokens=False)
    assert tokens.count(marker)==1;cut=tokens.index(marker);assert len(tokens)-cut==4
    return tokens[:cut],tokens[cut:]
def same(trial,a,b):
    for i,layer in enumerate(trial.model.model.layers):
        if layer.is_linear:pairs=[(a[i][0],b[i][0]),(a[i][1],b[i][1])]
        else:
            assert a[i].offset==b[i].offset
            n=a[i].offset;pairs=[(a[i].keys[:,:,:n],b[i].keys[:,:,:n]),(a[i].values[:,:,:n],b[i].values[:,:,:n])]
        assert all(bool(mx.array_equal(x,y).item()) for x,y in pairs),i
def size(trial,c):
    total=0
    for i,layer in enumerate(trial.model.model.layers):
        if layer.is_linear:total+=sum(v.nbytes for v in c[i].cache if v is not None)
        elif c[i].keys is not None:total+=c[i].keys[:,:,:c[i].offset,:].nbytes+c[i].values[:,:,:c[i].offset,:].nbytes
    return total
def measure(trial,x,key,cfg,observe,arms,limit):
    tick=elapsed();old_text=body(x,key,x['old'],cfg);new_text=body(x,key,x['new'],cfg)
    old_raw=old_text.encode();new_raw=new_text.encode();old_ids,tail=tokenize(trial,old_text)
    saved={0:make_prompt_cache(trial.model)};c=make_prompt_cache(trial.model)
    for i in range(0,len(old_ids),32):
        group=old_ids[i:i+32];advance(trial.model,c,group);saved[i+len(group)]=copy.deepcopy(c)
    old=c;old_s=elapsed()-tick;old_bytes=sum(size(trial,v) for v in saved.values())
    rows=[]
    for arm in arms:
        start=elapsed();donor=None;cost={};parts={}
        if arm in ['patch_full','patch_prefix']:
            blob=patch(old_raw,new_raw);packed=elapsed();packet,wire=send_local(blob);sent=elapsed()
            updated=apply_patch(old_raw,packet);assert updated==new_raw
            ids,tail2=tokenize(trial,updated.decode());common=prefix(old_ids,ids)
            if arm=='patch_prefix':client,cut=copy_at(saved,common)
            else:client,cut=make_prompt_cache(trial.model),0
            advance(trial.model,client,ids[cut:]);ready=elapsed()
            cost=dict(sender_s=packed-start,network_s=sent-packed,receiver_s=ready-sent,donor_s=0.0)
        else:
            ids,tail2=tokenize(trial,new_text);common=prefix(old_ids,ids)
            donor,cut=copy_at(saved,common);advance(trial.model,donor,ids[cut:]);made=elapsed()
            if arm=='answer_only':
                tokens=answer(trial,donor,tail2,limit)
                blob=np.asarray(tokens,dtype='<u4').tobytes();packed=elapsed();packet,wire=send_local(blob);sent=elapsed()
                assert list(np.frombuffer(packet,dtype='<u4'))==tokens
                cost=dict(sender_s=packed-made,network_s=sent-packed,receiver_s=elapsed()-sent,donor_s=made-start)
            else:
                blob,parts=encode(trial,old,donor,tail2,common,arm,observe);packed=elapsed()
                packet,wire=send_local(blob);sent=elapsed()
                client=recover(trial,old,packet,arm);ready=elapsed()
                cost=dict(sender_s=packed-made,network_s=sent-packed,receiver_s=ready-sent,donor_s=made-start)
        assert tail==tail2
        if arm=='answer_only':decode_s=0.0
        else:
            tokens=answer(trial,client,tail,limit);end=elapsed();decode_s=end-ready
        cold=sum(cost.values())+decode_s
        rows.append(dict(mode=arm,wire_bytes=wire,array_bytes=parts,old_tokens=len(old_ids),new_tokens=len(ids),common_tokens=common,replayed_from=cut,**cost,decode_s=decode_s,cold_s=cold,warm_s=cold-cost['donor_s'],tokens=tokens,text=trial.tok.decode(tokens)))
        if arm=='dense':same(trial,donor,client)
        mx.clear_cache()
    return dict(source=x['source'],id=x['id'],role=('changed' if key=='q' else 'stable'),old_setup_s=old_s,logical_snapshot_bytes=old_bytes,arms=rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('public',type=Path);a=ap.parse_args()
    d=a.public.resolve()/'experiments/read_projection';sys.path.insert(0,str(d))
    from run import Reader
    from codec import observability
    cfg=json.loads((d/'release.json').read_text());data=d/'data/dev_inputs.json'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==cfg['inputs']['dev']
    for n,h in cfg['model_files'].items():
        with (a.model/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    inputs=json.loads(data.read_text())
    cases=[v for src in ['babi','mquake'] for v in sorted([x for x in inputs if x['source']==src],key=lambda x:hashlib.sha256(('service0/'+src+'/'+str(x['id'])).encode()).digest())[:4]]
    assert len(cases)==8
    trial=Reader(a.model);mx.set_cache_limit(128*1024**2)
    warm_ids,warm_tail=tokenize(trial,'Give the number of days in a week.');warm=advance(trial.model,make_prompt_cache(trial.model),warm_ids);answer(trial,warm,warm_tail,3)
    rng=random.Random(9831);tasks=[(x,k) for x in cases for k in ['q','q_control']];rng.shuffle(tasks)
    path=HERE/'rows.jsonl';seen={ (v['source'],v['id'],v['role']) for v in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    begin=elapsed()
    with path.open('a') as f:
        for x,k in tasks:
            modes=['patch_full','patch_prefix','dense','projection','quantized','answer_only'];rng.shuffle(modes)
            if (x['source'],x['id'],'changed' if k=='q' else 'stable') in seen:continue
            v=measure(trial,x,k,cfg,observability,modes,12 if x['source']=='babi' else 24)
            f.write(json.dumps(v)+'\n');f.flush();print(x['source'],x['id'],k,round(elapsed()-begin,2),flush=True)
            if elapsed()-begin>1200:raise TimeoutError('Serving pilot budget')
if __name__=='__main__':main()
