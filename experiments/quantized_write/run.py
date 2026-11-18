"""Replay a genuine local 9B intervention on prospectively fixed new seeds."""
import copy
import ctypes
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import compute_g

p=Path(__file__).resolve().parent
from native_group_reader_math_v1 import optimize
from factor import starts,unit
from generator import make_example

def read(path):
    return json.loads(path.read_text())

def host(a):
    return np.ascontiguousarray(np.asarray(a.astype(mx.float32)),dtype=np.float64)

class Trial:
    def __init__(self,model_path,cfg):
        self.model,self.tok=load(str(model_path))
        layers=self.model.model.layers
        assert len(layers)==32 and layers[28].is_linear
        self.ordinal=[i for i,layer in enumerate(layers) if layer.is_linear].index(28)
        self.lo=2*layers[28].linear_attn.key_dim+28*128
        assert self.lo==7680
        self.kernel=qwen.gated_delta_update
        self.arc=ctypes.CDLL(str(p/'build/arc.dylib'))
        self.ref=ctypes.CDLL(str(p/'build/limited.dylib'))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.arc.arc_solve.argtypes=[ctypes.c_int]*5+[ptr]*6
        self.ref.solve_limited.argtypes=[ctypes.c_int]*5+[ptr]*5

    def encode(self,e,side):
        body=e['prefix_'+side]+e['suffix']
        full=self.tok.apply_chat_template([dict(role='user',content=body)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.tok.encode(full,add_special_tokens=False)
        op=self.tok.encode('<think>',add_special_tokens=False)
        assert len(op)==1 and ids.count(op[0])==1
        cut=ids.index(op[0]);assert len(ids)-cut==4
        return ids[:cut],ids[cut:]

    def prefill(self,ids):
        c=make_prompt_cache(self.model);self.model(mx.array([ids]),cache=c);mx.eval(*[layer.state for layer in c]);return c

    def write(self,c,donor):
        clone=copy.deepcopy(c)
        orig=clone[28][0]
        source=donor[28][0]
        new=mx.concatenate((orig[:,-1:,:self.lo],source[:,-1:,self.lo:self.lo+256],orig[:,-1:,self.lo+256:]),axis=-1)
        clone[28][0]=mx.concatenate((orig[:,:-1,:],new),axis=1)
        mx.eval(clone[28][0])
        return clone

    def steps(self,cache,suffix,overrides=None,capture=False):
        c=copy.deepcopy(cache);seen=[]
        for j,tok in enumerate(suffix):
            order=-1
            def hook(q,k,v,a,b,al,bias,state,mask,**kw):
                nonlocal order
                order+=1
                if order!=self.ordinal:
                    return self.kernel(q,k,v,a,b,al,bias,state,mask,**kw)
                if overrides is not None:
                    q=mx.concatenate((q[:,:,:14],mx.array(overrides[j],dtype=q.dtype)[None,None,None,:],q[:,:,15:]),axis=2)
                out,new=self.kernel(q,k,v,a,b,al,bias,state,mask,**kw)
                if capture:
                    seen.append(dict(n=q[0,0,14],key=k[0,0,14],v=v[0,0,28:30],beta=mx.sigmoid(b)[0,0,28:30],
                                     gate=compute_g(al,a,bias)[0,0,28:30],s=new[0,28:30].transpose(0,2,1)))
                return out,new
            qwen.gated_delta_update=hook
            try:
                logits=self.model(mx.array([[tok]]),cache=c)[0,-1]
                mx.eval(logits)
            finally:
                qwen.gated_delta_update=self.kernel
        return host(logits),[{key:host(a) for key,a in v.items()} for v in seen]

    def geometry(self,base,edit,suffix):
        logits0,b=self.steps(base,suffix,capture=True)
        logits1,e=self.steps(edit,suffix,capture=True)
        factors=[(np.empty((128,0)),np.empty((128,0))) for _ in range(2)]
        slots=[];rel=[]
        for a,z in zip(b,e,strict=True):
            for k in ('n','key','beta','gate'):
                assert np.array_equal(a[k],z[k]),k
            for h in range(2):
                r,u=factors[h];key=a['key'];beta=a['beta'][h];g=a['gate'][h]
                r=g*(r-beta*key[:,None]*(key@r)[None,:])
                dv=z['v'][h]-a['v'][h]
                if np.any(dv):
                    r=np.column_stack((r,key));u=np.column_stack((u,beta*dv))
                factors[h]=r,u
            d=np.ascontiguousarray(np.stack([r@u.T for r,u in factors]))
            live=z['s']-a['s']
            rel.append(float(np.linalg.norm(d-live)/max(np.linalg.norm(d),np.linalg.norm(live),1e-8)))
            slots.append((a['s'],d,a['n'],list(factors)))
        return logits0,logits1,slots,rel

    def solve(self,slot,mode):
        s,d,n,parts=slot
        if mode in ('white','ref80'):
            v=starts(s,parts,n,n,'original')
            if mode=='white':
                return v[2]*np.linalg.norm(n)
            def normalized(x):
                return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
            target=unit(normalized(n@(s+d))-normalized(n@s))
            return optimize(s,d,n,target,list(v))
        out=np.empty_like(n)
        if mode=='arc6':
            ret=self.arc.arc_solve(1,6,*s.shape,s,d,n,n,out,np.empty(2))
        else:
            ret=self.ref.solve_limited(16 if mode=='ref16' else 6,1,*s.shape,s,d,n,n,out)
        assert ret==0
        return out

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('model_path',type=Path)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args()
    evidence=p.parents[1]/'evidence/quantized_9b'
    cfg=read(evidence/'contract.json');path=args.model_path
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=read(p.parent/'read_projection/release.json')
    assert manifest['model']==cfg['model'] and manifest['revision']==cfg['revision']
    for f,h in manifest['model_files'].items():
        with (path/f).open('rb') as fp:assert hashlib.file_digest(fp,'sha256').hexdigest()==h,f
    modelcfg=read(path/'config.json')
    assert modelcfg['quantization']['bits']==4
    trial=Trial(path,cfg);make=make_example
    fixture=read(evidence/'fixture.json')
    e=make(fixture['example']['seed'],fixture['example']['surface'],fixture['example']['control'])
    assert e==fixture['example']
    for side in ('a','b'):
        head,tail=trial.encode(e,side)
        assert head+tail==fixture['tokens_'+side] and len(head)==fixture['cut']
    start=time.perf_counter()
    out_file=args.output_dir/'rows.jsonl'
    with out_file.open('x') as out:
        for i,seed in enumerate(cfg['seeds']):
            for surface in ('table','prose'):
                for nuisance in (False,True):
                    timer=time.perf_counter();e=make(seed,surface,nuisance)
                    a,tail=trial.encode(e,'a');b,tail2=trial.encode(e,'b')
                    assert tail==tail2 and len(a)==len(b)
                    na=trial.prefill(a);donor=trial.prefill(b);edit=trial.write(na,donor)
                    prefix_s=time.perf_counter()-timer
                    td=time.perf_counter();base,written,geom,errs=trial.geometry(na,edit,tail)
                    geom_s=time.perf_counter()-td
                    ids=[trial.tok.encode(e[v],add_special_tokens=False) for v in ('value_a','value_b')]
                    assert all(len(x)==1 for x in ids)
                    ia,ib=[x[0] for x in ids]
                    modes=cfg['modes'].copy();random.Random(seed+int(nuisance)+int(surface=='prose')*10).shuffle(modes)
                    for m in modes:
                        t=time.perf_counter()
                        plans=None if m=='native' else [trial.solve(slot,m) for slot in geom]
                        design=time.perf_counter()-t
                        for active,cache in [(False,na),(True,edit)]:
                            timer=time.perf_counter()
                            logits,_=trial.steps(cache,tail,plans)
                            if m=='native':
                                assert np.array_equal(logits,written if active else base)
                            gold=ib if active and not nuisance else ia
                            out.write(json.dumps(dict(seed=seed,surface=surface,nuisance=nuisance,written=active,mode=m,
                                gold=gold,top=int(logits.argmax()),margin=float(logits[ib]-logits[ia]),base_id=ia,donor_id=ib,
                                prefix_s=prefix_s,geometry_s=geom_s,design_s=design,rollout_s=time.perf_counter()-timer,
                                forecast_rel=errs,nonzero_ranks=[[int(u.shape[1]) for _,u in slot[-1]] for slot in geom]))+'\n')
                        out.flush()
            print('seeds',i+1,round(time.perf_counter()-start,1),flush=True)
    (args.output_dir/'receipt.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-start,model=manifest['model'],revision=manifest['revision']))+'\n')

if __name__=='__main__':
    main()
