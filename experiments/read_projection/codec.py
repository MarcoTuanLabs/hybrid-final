"""Encode a correction to a recipient's recurrent matrix in an observable subspace."""
import numpy as np

def observability(steps,heads):
    # For a state V x K, D_t = D_0 A_1 ... A_t where A_i=g_i(I-beta_i k_i k_i^T).
    # Each head's scalar product of retention gates scales but cannot rotate its read.
    qs=[];ks=[];bs=[];ws=[]
    for step in steps:
        q,k,b=step
        repeat=heads//len(q);q=np.repeat(q,repeat,axis=0);k=np.repeat(k,repeat,axis=0)
        qs.append(q);ks.append(k);bs.append(b)
        w=q.copy()
        for key,beta in reversed(list(zip(ks,bs))):
            w-=key*(beta*np.sum(key*w,axis=-1))[:,None]
        ws.append(w)
    return np.stack(qs,axis=-1),np.stack(ws,axis=-1)

def encode_delta(delta,mode,queries,transported):
    """All approximations are decoded using the transmitted float arrays or packed bytes."""
    if mode in ('raw_queries','transported'):
        columns=queries if mode=='raw_queries' else transported
        q=np.linalg.qr(columns,mode='reduced')[0].astype(np.float32)
        b=(delta@q).astype(np.float32)
        decoded=b@q.transpose(0,2,1)
        return decoded,q.nbytes+b.nbytes
    if mode=='svd':
        u,s,vt=np.linalg.svd(delta,full_matrices=False)
        rank=queries.shape[-1]
        b=np.ascontiguousarray((u[:,:,:rank]*s[:,None,:rank]).astype(np.float32))
        qt=np.ascontiguousarray(vt[:,:rank].astype(np.float32))
        return b@qt,b.nbytes+qt.nbytes
    if mode=='row_2bit':
        low=delta.min(axis=2,keepdims=True).astype(np.float16)
        scale=((delta.max(axis=2,keepdims=True)-low.astype(np.float32))/3).astype(np.float16)
        flo=low.astype(np.float32);fscale=scale.astype(np.float32)
        code=np.rint((delta-flo)/np.where(fscale>0,fscale,1)).clip(0,3).astype(np.uint8)
        packed=code[:,:,0::4]|(code[:,:,1::4]<<2)|(code[:,:,2::4]<<4)|(code[:,:,3::4]<<6)
        restored=np.stack([(packed>>shift)&3 for shift in (0,2,4,6)],axis=-1).reshape(delta.shape)
        return flo+restored.astype(np.float32)*fscale,packed.nbytes+low.nbytes+scale.nbytes
    raise ValueError(mode)
