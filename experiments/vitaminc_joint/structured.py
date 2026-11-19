"""Compare revision-pair readers using the same old and revised evidence."""
import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path

import numpy as np
from pairs import HERE
from score import ridge,flat

TYPES=list(combinations_with_replacement(range(3),2))


def decompose(x):
    return x.mean(1),(x[:,0]-x[:,1])/2


def standardize(train,test):
    offset=train.mean(0)
    rms=np.sqrt(np.mean((train-offset)**2)).clip(1e-8)
    return (train-offset)/rms,(test-offset)/rms


def readouts(train,test,gold):
    """The unordered label-pair posterior and conditional orientation yield endpoint marginals."""
    xmean,xdiff=decompose(train);zmean,zdiff=decompose(test)
    u,v=standardize(xmean,zmean)
    absu,absv=standardize(np.abs(xdiff),np.abs(zdiff))
    even=np.concatenate((u,absu),axis=1);even_test=np.concatenate((v,absv),axis=1)
    group=np.array([TYPES.index(tuple(sorted(labels))) for labels in gold])
    type_raw=ridge(even,np.eye(6)[group])(even_test)
    type_prob=np.maximum(type_raw,1e-8)
    type_prob/=type_prob.sum(-1,keepdims=True)
    result=np.zeros((len(test),2,3))
    for i,(a,b) in enumerate(TYPES):
        mass=type_prob[:,i]
        if a==b:
            result[:,:,a]+=mass[:,None]
            continue
        mask=group==i
        x=xdiff[mask];y=np.where(gold[mask,0]==a,1.,-1.)[:,None]
        # Pair reversal must reverse the predicted orientation.
        f=ridge(np.concatenate((x,-x)),np.concatenate((y,-y)),False)
        p=np.clip((1+f(zdiff)[:,0])/2,0,1)
        result[:,0,a]+=mass*p;result[:,1,a]+=mass*(1-p)
        result[:,0,b]+=mass*(1-p);result[:,1,b]+=mass*p
    assert np.allclose(result.sum(-1),1)
    y=np.eye(3)[gold];ym,yd=decompose(y)
    mean=ridge(xmean,ym)(zmean)
    slope=ridge(np.concatenate((xdiff,-xdiff)),np.concatenate((yd,-yd)),False)(zdiff)
    direct=ridge(flat(train),flat(y))(test)
    components=np.stack((mean+slope,mean-slope),axis=1)
    return dict(structured=result,direct=direct,components=components)


def load_q(cache,items,part):
    out=[]
    for item in items:
        with np.load(cache/part/(item["id"]+".npz")) as f:
            out.append({k:f[k] for k in f.files})
    return {k:np.stack([x[k] for x in out]) for k in out[0]}


def load_special(path,items,gold):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert [r['id'] for r in rows]==[r['id'] for r in items]
    assert np.array_equal([r['gold'] for r in rows],gold)
    return np.array([r['logits'] for r in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument("cache",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();sp=json.loads((HERE/'split.json').read_text())
    tr=load_q(a.cache,sp["fit"],"fit");ev=load_q(a.cache,sp["pilot"],"pilot")
    g=ev["gold"]
    methods=readouts(tr["h"],ev["h"],tr["gold"])
    rev=readouts(tr["h"],ev["h"][:,::-1],tr["gold"])
    for k in methods:assert np.allclose(methods[k],rev[k][:,::-1],atol=1e-7)
    full=ev["full"]
    methods.update(full=full,previous=full[:,::-1],contrast=2*full-full[:,::-1],explicit=ev["explicit"])
    for model in ("albert-base-vitaminc-mnli","albert-xlarge-vitaminc"):
        fit=load_special(a.output/(model+"-fit.jsonl"),sp["fit"],tr["gold"])
        test=load_special(a.output/(model+"-pilot.jsonl"),sp["pilot"],g)
        methods[model]=test
        for k,z in readouts(fit,test,tr["gold"]).items():methods[model+"_"+k]=z
    pred={k:z.argmax(-1) for k,z in methods.items()}
    stable=g[:,0]==g[:,1]
    results={k:dict(correct=int((z==g).sum()),stable=int((z[stable]==g[stable]).sum()),
                    changed=int((z[~stable]==g[~stable]).sum()),both=int((z==g).all(1).sum()),
                    wins_vs_primary=int(((z==g)&(pred['structured']!=g)).sum()),
                    losses_vs_primary=int(((z!=g)&(pred['structured']==g)).sum())) for k,z in pred.items()}
    best=max(z['correct'] for k,z in results.items() if k!='structured')
    gate=results['structured']['correct']>=best+4 and results['structured']['stable']>=results['direct']['stable']-1
    res=dict(pages=len(g),endpoints=g.size,stable=int(2*stable.sum()),fitted_pages=len(tr['h']),
             label_counts=np.bincount(g.flatten(),minlength=3).tolist(),methods=results,promote=gate)
    (a.output/'summary.json').write_text(json.dumps(res,indent=2)+'\n')
    paired=[dict(id=r['id'],page=r['page'],gold=g[i].tolist(),outputs={k:z[i].tolist() for k,z in pred.items()}) for i,r in enumerate(sp['pilot'])]
    (a.output/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in paired))
    print(json.dumps(res))
if __name__=='__main__':main()
