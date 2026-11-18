"""Check both block-deletion rules against direct masked single-token equations."""
import json
from pathlib import Path
import mlx.core as mx
from mlx_lm.models import gated_delta
import numpy as np

P=Path(__file__).resolve().parent
mx.set_default_device(mx.cpu)
rng=np.random.default_rng(99121)
maxerr={'skip':0.,'values':0.}
changed=0
for rep in (1,2):
    for lo,hi in ((0,3),(3,8),(8,12)):
        n=12;hk=2;hv=hk*rep;d=32;vwidth=8
        raw=lambda *shape:rng.normal(size=shape).astype('float32')
        keys=raw(1,n,hk,d);keys/=np.linalg.norm(keys,axis=-1,keepdims=True)
        q=raw(1,n,hk,d);v=raw(1,n,hv,vwidth)
        a=raw(1,n,hv);b=raw(1,n,hv);al=raw(hv)-2;bias=raw(hv)
        g=np.asarray(gated_delta.compute_g(mx.array(al),mx.array(a),mx.array(bias)))
        beta=1/(1+np.exp(-b))
        for mode in maxerr:
            initial=raw(1,hv,vwidth,d);st=mx.array(initial)
            for l,r in [(0,lo),(lo,hi),(hi,n)]:
                if l==r or (l==lo and r==hi and mode=='skip'):continue
                val=v[:,l:r] if (l,r)!=(lo,hi) else np.zeros_like(v[:,l:r])
                _,st=gated_delta.gated_delta_update(*map(mx.array,[q[:,l:r],keys[:,l:r],val,a[:,l:r],b[:,l:r],al,bias]),st,None,use_kernel=False)
            ref=initial.copy()
            for t in range(n):
                if lo<=t<hi and mode=='skip':continue
                k=np.repeat(keys[0,t],rep,axis=0)
                ref*=g[:,t,:,None,None]
                val=v[:,t] if not lo<=t<hi else np.zeros_like(v[:,t])
                err=val-(ref*k[None,:,None,:]).sum(-1)
                ref+=(beta[:,t,:,None]*err)[...,None]*k[None,:,None,:]
            maxerr[mode]=max(maxerr[mode],float(np.max(np.abs(np.asarray(st)-ref))))
            changed+=1
assert max(maxerr.values())<2e-6,maxerr
out=dict(cases=changed,max_absolute_state_error=maxerr,condition='Fixed observed keys, values and retention; includes prefix, internal and suffix deletions, grouped as well as ungrouped heads.')
(P/'algebra.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
