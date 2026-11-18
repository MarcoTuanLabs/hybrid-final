"""Does a shared or batched prefix preserve the actual sparse write and save a complete service?"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask,create_ssm_mask
from mlx_lm.models.cache import KVCache,make_prompt_cache
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'jacobian_service'))
from compare import cut_donor,set_write
def pref(trial,ids,c=None):
    if c is None:c=make_prompt_cache(trial.model)
    if ids:
        trial.model.model(mx.array([ids]),cache=c);mx.eval(*[l.state for l in c])
    return c
def cut_from(trial,remaining,common):
    net=trial.model.model;c=copy.deepcopy(common);h=net.embed_tokens(mx.array([remaining]))
    fa=create_attention_mask(h,c[net.fa_idx]);ssm=create_ssm_mask(h,c[net.ssm_idx])
    for i,l in enumerate(net.layers[:28]):h=l(h,mask=ssm if l.is_linear else fa,cache=c[i])
    l=net.layers[28];raw=l.linear_attn.in_proj_qkv(l.input_layernorm(h))
    return raw[:,-1:,trial.lo:trial.lo+256]
def branch(trial,a,b,mode):
    if mode=='shared':
        n=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y);common=pref(trial,a[:n])
        donor=cut_from(trial,b[n:],common);base=pref(trial,a[n:],copy.deepcopy(common))
    elif mode=='joint':
        c=make_prompt_cache(trial.model);trial.model.model(mx.array([a,b]),cache=c);mx.eval(*[k.state for k in c])
        donor=c[28][0][1:2,-1:,trial.lo:trial.lo+256]
        base=[]
        for layer in c:
            if isinstance(layer,KVCache):
                recipient=KVCache()
                recipient.state=(layer.keys[:1],layer.values[:1],layer.offset)
            else:
                recipient=layer.extract(0)
            base.append(recipient)
    else:
        base=trial.prefill(a);donor=cut_donor(trial,b) if mode=='cut' else trial.prefill(b)[28][0][:,-1:,trial.lo:trial.lo+256]
    return base,donor
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('public',type=Path);a=ap.parse_args()
    repo=a.public.resolve();sys.path.insert(0,str(repo/'experiments/quantized_write'))
    from run import Trial,make_example
    runtime_dir=HERE.parent/'jacobian_service' if (HERE.parent/'jacobian_service/runtime.py').exists() else HERE.parent/'stream9'
    sys.path.insert(0,str(runtime_dir));from runtime import stream
    manifest=json.loads((repo/'experiments/read_projection/release.json').read_text())
    for n,h in manifest['model_files'].items():
        with (a.model/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    trial=Trial(a.model,json.loads((repo/'evidence/quantized_9b/contract.json').read_text()))
    modes=['full','cut','shared','joint']
    def run(ex,m,active):
        mx.synchronize();t=time.perf_counter();a,tail=trial.encode(ex,'a');b,other=trial.encode(ex,'b');assert tail==other and len(a)==len(b)
        base,donor=branch(trial,a,b,m);edit=set_write(trial,base,donor);prefix_s=time.perf_counter()-t
        logits,q=stream(trial,base,edit,tail,active,'batch_observed');top=int(np.argmax(logits));mx.synchronize();wall=time.perf_counter()-t
        before=base[28][0][:,-1:,trial.lo:trial.lo+256]
        return dict(top=top,seconds=wall,prefix_s=prefix_s,query_hash=hashlib.sha256(np.asarray(q).tobytes()).hexdigest()),np.asarray(donor.astype(mx.float32)),np.asarray(before.astype(mx.float32)),logits
    for m in modes:run(make_example(27112000,'table',False),m,True)
    start=time.monotonic();out=HERE/'rows.jsonl'
    seen={tuple(row['key']) for row in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
    with out.open('a') as f:
        for seed in range(27112000,27112008):
            for surf in ['table','prose']:
                for nuisance in [False,True]:
                    ex=make_example(seed,surf,nuisance)
                    for active in [False,True]:
                        key=(seed,surf,nuisance,active)
                        if key in seen:continue
                        order=modes.copy();random.Random(seed*41+int(nuisance)*7+int(active)*11+int(surf=='table')).shuffle(order)
                        results={m:run(ex,m,active) for m in order};ref=results['full'];arms={}
                        for m,(row,write,before,lg) in results.items():
                            arms[m]=row|dict(differing_write_values=int(np.count_nonzero(write.view(np.uint32)!=ref[1].view(np.uint32))),max_write_diff=float(np.max(abs(write-ref[1]))),different_recipient_values=int(np.count_nonzero(before.view(np.uint32)!=ref[2].view(np.uint32))),max_logit_diff=float(np.max(abs(lg-ref[3]))))
                        ids=[trial.tok.encode(ex[k],add_special_tokens=False)[0] for k in ['value_a','value_b']]
                        f.write(json.dumps(dict(key=key,recipient=ids[0],donor=ids[1],arms=arms))+'\n');f.flush()
            print(seed,round(time.monotonic()-start,1),flush=True);mx.clear_cache()
            if time.monotonic()-start>600:raise TimeoutError('Prefix service budget')
if __name__=='__main__':main()
