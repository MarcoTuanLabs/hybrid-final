"""Test an exact partial donor and a cone-constrained one-step Jacobian reader."""
import argparse
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import create_attention_mask,create_ssm_mask
import numpy as np
HERE=Path(__file__).resolve().parent

def cut_donor(trial,ids):
    net=trial.model.model;cache=make_prompt_cache(trial.model);h=net.embed_tokens(mx.array([ids]))
    fa=create_attention_mask(h,cache[net.fa_idx]);ssm=create_ssm_mask(h,cache[net.ssm_idx])
    for i,layer in enumerate(net.layers[:28]):h=layer(h,mask=ssm if layer.is_linear else fa,cache=cache[i])
    layer=net.layers[28];raw=layer.linear_attn.in_proj_qkv(layer.input_layernorm(h))
    return raw[:,-1:,trial.lo:trial.lo+256]
def set_write(trial,base,donor_slice):
    clone=copy.deepcopy(base);o=clone[28][0];l=trial.lo
    last=mx.concatenate([o[:,-1:,:l],donor_slice,o[:,-1:,l+256:]],axis=-1)
    clone[28][0]=mx.concatenate([o[:,:-1,:],last],axis=1);mx.eval(clone[28][0]);return clone
def jacobian(slot):
    s,d,n,_=slot;norm=np.linalg.norm(n);unit=n/norm;mat=s+d
    def scale(x):return np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
    def read(matrix,q):return q@matrix
    x=read(s,n);y=read(mat,n);rx=scale(x);ry=scale(y);nx=x/rx;ny=y/ry
    diff=ny-nx;z=np.linalg.norm(diff)
    if z<1e-15:return n.copy()
    target=diff/z
    def pull(m,normalized,rho):
        t=(target-normalized*np.mean(target*normalized,axis=-1,keepdims=True))/rho
        return np.einsum('hkv,hv->k',m,t)
    grad=pull(mat,ny,ry)-pull(s,nx,rx);g=np.linalg.norm(grad)
    if g<1e-15:return n.copy()
    direction=grad/g;dot=direction@unit
    if dot<.5:
        tangent=direction-dot*unit;length=np.linalg.norm(tangent)
        if length<1e-15:return n.copy()
        direction=.5*unit+np.sqrt(.75)*tangent/length
    q=norm*direction
    x2=read(s,q);y2=read(mat,q)
    return q if np.sum((y2/scale(y2)-x2/scale(x2))*target)>z else n.copy()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('public',type=Path);a=ap.parse_args()
    repo=a.public.resolve();sys.path.insert(0,str(repo/'experiments/quantized_write'))
    from run import Trial,make_example
    runtime=HERE/'runtime.py'
    if runtime.exists():sys.path.insert(0,str(HERE))
    else:sys.path.insert(0,str(HERE.parent/'stream9'))
    from runtime import stream
    cfg=json.loads((repo/'evidence/quantized_9b/contract.json').read_text());pins=json.loads((repo/'experiments/read_projection/release.json').read_text())
    for f,h in pins['model_files'].items():
        with (a.model/f).open('rb') as fp:assert hashlib.file_digest(fp,'sha256').hexdigest()==h
    trial=Trial(a.model,cfg);original=trial.solve
    def service(ex,mode,active):
        mx.synchronize();t=time.perf_counter()
        ids,tail=trial.encode(ex,'a');alternate,tail2=trial.encode(ex,'b');assert tail==tail2
        base=trial.prefill(ids);donor=(trial.prefill(alternate)[28][0][:,-1:,trial.lo:trial.lo+256] if mode=='full16' else cut_donor(trial,alternate))
        edit=set_write(trial,base,donor);split=time.perf_counter();native=[];raw_chosen=[]
        def solve(slot,name):
            s,d,n,_=slot;native.append(n)
            if mode=='cut_jacobian':ans=jacobian(slot)
            elif mode=='cut4':
                ans=np.empty_like(n);ret=trial.ref.solve_limited(4,1,*s.shape,s,d,n,n,ans);assert ret==0
            else:ans=original(slot,name)
            raw_chosen.append(ans);return ans
        trial.solve=solve
        try:logits,qs=stream(trial,base,edit,tail,active,'batch_native' if mode=='cut_native' else 'batch_observed')
        finally:trial.solve=original
        top=int(np.argmax(logits));mx.synchronize();end=time.perf_counter()
        ratio=[];cos=[]
        for orig,new in zip(native,qs,strict=True):
            ratio.append(np.linalg.norm(new)/np.linalg.norm(orig));cos.append(np.dot(orig,new)/(np.linalg.norm(orig)*np.linalg.norm(new)))
        if ratio:assert max(abs(1-v) for v in ratio)<.01 and min(cos)>=.49
        return dict(top=top,seconds=end-t,prefill_write_s=split-t,query_and_output_s=end-split,donor_digest=hashlib.sha256(np.asarray(donor.view(mx.uint16)).tobytes()).hexdigest(),query_digest=hashlib.sha256(np.asarray(qs).tobytes()).hexdigest(),norm_ratios=ratio,cosines=cos)
    modes=['full16','cut16','cut4','cut_jacobian','cut_native'];seed_values=range(27112000,27112008)
    for mode in modes:service(make_example(27112000,'table',False),mode,True)
    dest=HERE/'rows.jsonl';seen={tuple(v['key']) for v in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    start=time.monotonic()
    with dest.open('a') as f:
        for seed in seed_values:
            for surface in ['table','prose']:
                for nuisance in [False,True]:
                    ex=make_example(seed,surface,nuisance)
                    for active in [True,False]:
                        key=(seed,surface,nuisance,active)
                        if key in seen:continue
                        order=modes.copy();random.Random(seed*19+int(nuisance)*7+int(active)*5+int(surface=='prose')*101).shuffle(order)
                        row={m:service(ex,m,active) for m in order}
                        same_write=len(set(x['donor_digest'] for x in row.values()))==1
                        # If a truncation changes a write, subsequent timing cannot support an identical-write claim.
                        assert same_write,key
                        donor_id,recipient_id=[trial.tok.encode(ex[k],add_special_tokens=False)[0] for k in ['value_b','value_a']]
                        f.write(json.dumps(dict(key=key,arms=row,donor=donor_id,recipient=recipient_id))+'\n');f.flush()
            print(seed,round(time.monotonic()-start,1),flush=True);mx.clear_cache()
            if time.monotonic()-start>720:raise TimeoutError('Full-service budget')
if __name__=='__main__':main()
