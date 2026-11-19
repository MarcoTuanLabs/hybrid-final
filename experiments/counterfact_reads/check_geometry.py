import json
from pathlib import Path
import sys
import numpy as np
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r))
sys.path.insert(0,str(r.parent/'read_projection'))
from geometry import designs
from codec import observability
rng=np.random.default_rng(1947)
d=rng.normal(size=(32,128,128));w=rng.normal(size=(32,128,4));bg=rng.normal(size=(32,128,4))
tests={}
for method,i,a,n in designs(d,w,bg):
    assert not i%2 and i+1<32 and a.shape==(2,128,128)
    if method in ('normalized','raw','fixed'):
        error=np.linalg.norm((d[i:i+2]-a)@w[i:i+2],axis=(1,2))
        exact=np.linalg.svd(d[i:i+2]@w[i:i+2],compute_uv=False)
        opt=np.linalg.norm(exact[:,1:],axis=1)
        np.testing.assert_allclose(error,opt,rtol=1e-6)
    tests[method]=dict(selected_head=i,payload=n)
steps=[];g=[];state=d.copy();truth=[]
for t in range(4):
    q=rng.normal(size=(16,128));k=rng.normal(size=(16,128));k/=np.linalg.norm(k,axis=1,keepdims=True);b=rng.uniform(size=32)
    gates=rng.uniform(.6,1,size=32);g.append(gates);steps.append((q,k,b))
    key=np.repeat(k,2,axis=0)
    state=(state - ((state@key[:,:,None])*(b[:,None,None]*key[:,None,:])))*gates[:,None,None]
    truth.append((state@np.repeat(q,2,axis=0)[:,:,None]).squeeze(-1))
_,transport=observability(steps,32)
est=d@(transport*np.cumprod(np.stack(g,axis=-1),axis=-1)[:,None,:]);gold=np.stack(truth,axis=-1)
np.testing.assert_allclose(est,gold,atol=1e-10)
print('rank-one read-response optimality and transported recurrence verified',tests,'max error',np.abs(est-gold).max())
