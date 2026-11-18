"""Construct the fixed 9B readout starts from the carried key/value write factors."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve,cholesky,solve_triangular

def unit(v):
    return v/max(np.linalg.norm(v),1e-30)

def orient(v,n):
    v=unit(v)
    return -v if v@n<0 else v

def trace_steps(path):
    with np.load(path) as z:
        factors=[(np.empty((128,0)),np.empty((128,0))) for _ in range(2)]
        for t in range(4):
            states=[]
            for h,(r,u) in enumerate(factors):
                get=lambda key:z[f'{key}_h{h}_t{t}']
                k=get('key');beta=float(get('beta'));g=float(get('decay'));dv=get('delta_value')
                r=g*(r-beta*k[:,None]*(k@r)[None,:])
                if np.any(dv!=0):
                    r=np.column_stack((r,k))
                    # Reproduce the source expression beta * dv before conversion to double.
                    u=np.column_stack((u,beta*dv))
                factors[h]=r,u
                states.append(get('state').astype('float64'))
            n=z[f'query_effective_h0_t{t}'].astype(float)
            raw=z[f'query_raw_h0_t{t}']
            labels=[z[f'query_{key}_t{t}'].astype(float) for key in ('raw_svd','whitened_key','whitened_svd','group_normalized')]
            yield np.stack(states),list(factors),n,raw,labels

def starts(s,parts,n,raw,kind):
    k=len(n)
    key=parts[0][0][:,0] if parts[0][0].shape[1] else raw
    cov=sum(x@x.T for x in s)
    cov.flat[::k+1]+=max(1e-3*np.trace(cov)/k,1e-12)
    if kind in ('original','dense_chol'):
        d=np.concatenate([r@u.T for r,u in parts],axis=1)
        v,sing,_=np.linalg.svd(d,full_matrices=False)
        rawsvd=v[:,0] if sing[0] else n
        if kind=='original':
            eig,basis=np.linalg.eigh(cov)
            inv=(basis/np.sqrt(eig))@basis.T
            w=np.linalg.solve(cov,key)
            ws=inv@np.linalg.svd(inv@d,full_matrices=False)[0][:,0]
        else:
            L=cholesky(cov,lower=True,check_finite=False)
            w=cho_solve((L,True),key,check_finite=False)
            u=np.linalg.svd(solve_triangular(L,d,lower=True,check_finite=False),full_matrices=False)[0][:,0]
            ws=solve_triangular(L.T,u,lower=False,check_finite=False)
    else:
        # D D^T = F F^T. QR of each value factor avoids squaring its condition number.
        skinny=[r@np.linalg.qr(u,mode='r').T for r,u in parts if u.shape[1]]
        if not skinny:
            return starts(s,parts,n,raw,'original')
        f=np.concatenate(skinny,axis=1)
        if kind=='thin_svd':
            rawsvd=np.linalg.svd(f,full_matrices=False)[0][:,0]
            L=cholesky(cov,lower=True,check_finite=False)
            y=solve_triangular(L,f,lower=True,check_finite=False)
            a=np.linalg.svd(y,full_matrices=False)[0][:,0]
            w=cho_solve((L,True),key,check_finite=False)
            ws=solve_triangular(L.T,a,lower=False,check_finite=False)
        else:
            sol=cho_solve(cho_factor(cov,lower=True,check_finite=False),np.column_stack((key,f)),check_finite=False)
            w=sol[:,0];z=sol[:,1:]
            rawsvd=f@np.linalg.eigh(f.T@f)[1][:,-1]
            ws=z@np.linalg.eigh((f.T@z+z.T@f)*.5)[1][:,-1]
    return np.stack([unit(n),orient(rawsvd,n),orient(w,n),orient(ws,n)])

def bf16(x):
    a=np.ascontiguousarray(x,dtype='float32').view(np.uint32)
    return (a+np.uint32(0x7fff)+((a>>16)&1))>>16
