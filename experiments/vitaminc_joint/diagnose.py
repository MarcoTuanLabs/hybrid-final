"""Separate errors choosing the pair of verdicts from errors assigning their order."""
import argparse
import json
from pathlib import Path
import numpy as np
from structured import TYPES,decompose,standardize,load_q,readouts
from pairs import HERE
from score import ridge


def main():
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    sp=json.loads((HERE/'split.json').read_text())
    fit=load_q(a.cache,sp['fit'],'fit');test=load_q(a.cache,sp['pilot'],'pilot')
    x,dx=decompose(fit['h']);z,dz=decompose(test['h'])
    u,v=standardize(x,z);ua,va=standardize(np.abs(dx),np.abs(dz))
    grp=lambda gs:np.array([TYPES.index(tuple(sorted(g))) for g in gs])
    t=grp(fit['gold']);actual=grp(test['gold'])
    posterior=np.maximum(ridge(np.hstack([u,ua]),np.eye(6)[t])(np.hstack([v,va])),1e-8)
    posterior/=posterior.sum(axis=1,keepdims=True)
    mass=np.zeros((len(z),6,2,3))
    for i,(left,right) in enumerate(TYPES):
        if left==right:mass[:,i,:,left]=1;continue
        xs=dx[t==i];y=np.where(fit['gold'][t==i,0]==left,1.,-1.)[:,None]
        f=ridge(np.concatenate([xs,-xs]),np.concatenate([y,-y]),False)
        orient=np.clip((1+f(dz)[:,0])/2,0,1)
        mass[:,i,0,left]=orient;mass[:,i,1,left]=1-orient
        mass[:,i,1,right]=orient;mass[:,i,0,right]=1-orient
    marg=np.einsum('ni,nijk->njk',posterior,mass)
    assert np.allclose(marg,readouts(fit['h'],test['h'],fit['gold'])['structured'],atol=1e-7)
    gold=test['gold'];fixed=mass[np.arange(len(z)),actual].argmax(-1);mix=gold[:,0]!=gold[:,1]
    pred=marg.argmax(-1)
    raw=[json.loads(s) for s in (a.output/'rows.jsonl').read_text().splitlines()]
    assert np.array_equal(pred,[r['outputs']['structured'] for r in raw])
    top=posterior.argmax(1)
    labels=['supports','refutes','insufficient']
    per=[]
    for i,(l,r) in enumerate(TYPES):
        m=actual==i
        if np.any(m):
            per.append(dict(pair=[labels[l],labels[r]],pages=int(m.sum()),predicted_pair_correct=int((top[m]==i).sum()),actual_endpoint_correct=int((pred[m]==gold[m]).sum()),given_pair_endpoint_correct=int((fixed[m]==gold[m]).sum())))
    s=dict(pages=len(z),type_correct=int((top==actual).sum()),stable_or_changed_correct=int(((np.array([TYPES[k][0]!=TYPES[k][1] for k in top]))==mix).sum()),correct_if_unordered_gold_pair_supplied=int((fixed==gold).sum()),oriented_changed_pages_if_gold_pair_supplied=int((fixed[mix]==gold[mix]).all(1).sum()),changed_pages=int(mix.sum()),by_true_type=per,purpose="Post-hoc diagnostic with access to the true unordered label pair. The service has no such access.")
    (a.output/'diagnostic.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(s))
if __name__=='__main__':main()
