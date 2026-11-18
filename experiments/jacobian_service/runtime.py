"""Fold readout design into a single suffix pass at one recurrent layer."""
import copy
import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models.qwen3_5 as qwen
from mlx_lm.models.gated_delta import compute_g
import numpy as np
from scipy.linalg import cho_factor,cho_solve

from run import Trial,host

class AlternateValues:
    def __init__(self,conv,history,lo):
        self.conv=conv
        self.history=history[:,:,lo:lo+256]
        self.lo=lo
        self.values=None
    def __call__(self,x):
        new=x[:,self.history.shape[1]:,self.lo:self.lo+256]
        buf=mx.concatenate([self.history,new],axis=1)
        self.history=buf[:,-self.history.shape[1]:,:]
        weights=self.conv.weight[self.lo:self.lo+256]
        self.values=nn.silu(mx.conv1d(buf,weights,groups=256)).reshape(1,new.shape[1],2,128)
        return self.conv(x)

def white(s,key,n):
    cov=sum(x@x.T for x in s)
    cov.flat[::len(n)+1]+=max(1e-3*np.trace(cov)/len(n),1e-12)
    v=cho_solve(cho_factor(cov,lower=True,check_finite=False),key,check_finite=False)
    v*=np.linalg.norm(n)/max(np.linalg.norm(v),1e-30)
    return -v if v@n<0 else v

def stream(trial,base,edit,tail,active,mode):
    if mode=='batch_staged':
        _,plan=stream(trial,base,edit,tail,active,'plan')
        cache=copy.deepcopy(edit if active else base)
        calls=0
        def planned(q,k,v,a,b,al,bias,state,mask,**kw):
            nonlocal calls
            ix=calls;calls+=1
            if ix==trial.ordinal:
                override=mx.array(np.stack(plan),dtype=q.dtype)[None,:,None,:]
                q=mx.concatenate([q[:,:,:14],override,q[:,:,15:]],axis=2)
            return trial.kernel(q,k,v,a,b,al,bias,state,mask,**kw)
        qwen.gated_delta_update=planned
        try:
            logits=trial.model(mx.array([tail]),cache=cache)[0,-1]
            mx.eval(logits)
        finally:
            qwen.gated_delta_update=trial.kernel
        return host(logits),plan
    class PlanReady(Exception):pass
    c=copy.deepcopy(edit if active else base)
    layer=trial.model.model.layers[28].linear_attn
    conv=layer.conv1d
    shadow=AlternateValues(conv,(base if active else edit)[28][0],trial.lo)
    kernel=trial.kernel
    s0=base[28][1][:,28:30]
    s1=s0
    factors=[(np.empty((128,0)),np.empty((128,0))) for _ in range(2)]
    chosen=[]
    step=0
    def hook(q,k,v,a,b,al,bias,state,mask,**kw):
        nonlocal step,s0,s1
        ix=hook.ordinal
        hook.ordinal+=1
        if ix!=trial.ordinal:
            return kernel(q,k,v,a,b,al,bias,state,mask,**kw)
        assert mask is None or mask.ndim==2
        H=slice(28,30);Q=slice(14,15)
        physical=v[:,:,H];alt=shadow.values
        v0,v1=(alt,physical) if active else (physical,alt)
        native=host(q[0,:,14])
        if mode!='batch_observed':
            diff=host(v1)-host(v0);keys=host(k[0,:,14])
            betas=host(mx.sigmoid(b)[0,:,H]);gates=host(compute_g(al,a,bias)[0,:,H])
        local=[]
        for j in range(q.shape[1]):
            time_slice=slice(j,j+1)
            def advance(old,value):
                return kernel(q[:,time_slice,Q],k[:,time_slice,Q],value[:,time_slice],a[:,time_slice,H],b[:,time_slice,H],
                              al[H],bias[H],old,None if mask is None else mask[:,time_slice],**kw)[1]
            s0=advance(s0,v0)
            source=host(s0[0].transpose(0,2,1))
            if mode=='batch_observed':
                s1=advance(s1,v1)
                delta=host(s1[0].transpose(0,2,1))-source
            else:
                for h,(r,u) in enumerate(factors):
                    if mode=='batch_white' and h:continue
                    key=keys[j];beta=betas[j,h]
                    r=gates[j,h]*(r-beta*key[:,None]*(key@r)[None,:])
                    if np.any(diff[0,j,h]) and (mode!='batch_white' or r.shape[1]==0):
                        r=np.column_stack((r,key));u=np.column_stack((u,beta*diff[0,j,h]))
                    factors[h]=r,u
                if mode!='batch_white':
                    delta=np.ascontiguousarray(np.stack([r@u.T for r,u in factors]))
            if mode=='batch_white':
                r,_=factors[0];vec=white(source,r[:,0] if r.shape[1] else native[j],native[j])
            else:
                vec=trial.solve((source,delta,native[j],factors),'ref16')
            value=mx.array(vec,dtype=q.dtype)
            local.append(value);chosen.append(value)
            step+=1
        if mode=='plan':raise PlanReady
        override=mx.stack(local)[None,:,None,:]
        q=mx.concatenate([q[:,:,:14],override,q[:,:,15:]],axis=2)
        return kernel(q,k,v,a,b,al,bias,state,mask,**kw)
    batches=[[t] for t in tail] if mode=='serial_inline' else [tail]
    if mode=='batch_native':
        logits=trial.model(mx.array([tail]),cache=c)[0,-1]
        mx.eval(logits)
        return host(logits),[]
    layer.conv1d=shadow
    qwen.gated_delta_update=hook
    try:
        for seq in batches:
            hook.ordinal=0
            logits=trial.model(mx.array([seq]),cache=c)[0,-1]
            mx.eval(logits)
    except PlanReady:
        assert mode=='plan'
    finally:
        layer.conv1d=conv
        qwen.gated_delta_update=kernel
    assert step==len(tail)
    return None if mode=='plan' else host(logits),[host(q) for q in chosen]
