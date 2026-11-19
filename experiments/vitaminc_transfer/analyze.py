"""Evaluate the frozen specialist readout and paired controls on new pages."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from transfer import HERE
from structured import readouts
from score import ridge,prob


def read(path):return [json.loads(line) for line in path.read_text().splitlines()]


def predict(fit,y,new):
    m=readouts(fit,new,y)
    m['native']=new;m['contrast']=2*new-new[:,::-1]
    joint=(np.ones((3,3))*2)
    for a,b in y:joint[a,b]+=1;joint[b,a]+=1
    joint/=joint.sum();assoc=joint/np.outer(joint.sum(1),joint.sum(0))
    # Calibrate temperature using calibration pages alone.
    best=min((float(-np.log(np.take_along_axis(prob(fit/temp),y[:,:,None],axis=2).clip(1e-12)).mean()),temp) for temp in (0.5,1,2,4))[1]
    calibrated=prob(new/best)
    product=calibrated[:,:,None,:]*(calibrated[:,::-1,:,None]*assoc[None,None,:,:])
    m['transition_prior']=product.sum(2)
    # Generic nine-class model sees six logits and the same labeled pairs.
    def oriented(z):return np.concatenate([z,z[:,::-1]],axis=0)
    train=oriented(fit);labels=oriented(y);outcomes=np.eye(9)[3*labels[:,0]+labels[:,1]]
    f=ridge(train.reshape(len(train),6),outcomes)
    def score(z):
        s=f(z.reshape(len(z),6)).reshape(-1,3,3).clip(1e-8)
        return s/s.sum((1,2),keepdims=True)
    table=(score(new)+score(new[:,::-1]).transpose(0,2,1))/2
    m['nine_class']=np.stack([table.sum(2),table.sum(1)],axis=1)
    return {k:v.argmax(-1) for k,v in m.items()},best


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    d=HERE.parent/'vitaminc_joint'
    tr=read(d/'pilot/albert-xlarge-vitaminc-fit.jsonl')
    assert [r['id'] for r in tr]==[r['id'] for r in json.loads((d/'split.json').read_text())['fit']]
    ev=read(a.output/'logits.jsonl')
    assert [dict(id=r['id'],page=r['page']) for r in ev]==json.loads((HERE/'split.json').read_text())['items']
    x=np.array([r['logits'] for r in tr]);y=np.array([r['gold'] for r in tr])
    z=np.array([r['logits'] for r in ev]);g=np.array([r['gold'] for r in ev])
    assert x.shape==(240,2,3) and z.shape==(224,2,3) and np.isfinite(z).all()
    out,temp=predict(x,y,z);rev,_=predict(x,y,z[:,::-1])
    for k,ans in out.items():assert np.array_equal(ans[:,::-1],rev[k])
    stable=g[:,0]==g[:,1];base=out['structured']==g
    scores={k:dict(correct=int((v==g).sum()),changed=int((v[~stable]==g[~stable]).sum()),stable=int((v[stable]==g[stable]).sum()),
                   both=int((v==g).all(1).sum()),wins_for_structured=int((base&(v!=g)).sum()),
                   losses_for_structured=int((~base&(v==g)).sum())) for k,v in out.items()}
    names=[k for k in out if k!='structured']
    diffs=np.stack([(base.sum(1)-(out[k]==g).sum(1))/2 for k in names],axis=1)
    rng=np.random.default_rng(765218);reps=[]
    for _ in range(40):
        sample=rng.integers(0,len(ev),size=(500,len(ev)))
        reps.extend(diffs[sample].mean(1).min(1).tolist())
    ci=np.quantile(reps,[.025,.975]).tolist()
    margin=float(diffs.mean(0).min());ok=margin>=0.04 and ci[0]>0 and scores['structured']['stable']>=scores['native']['stable']-1
    result=dict(pages=len(ev),endpoints=g.size,changed=int((~stable).sum()*2),stable=int(stable.sum()*2),methods=scores,
                temperature=temp,minimum_paired_gain=margin,minimum_gain_interval=ci,confirmed=bool(ok))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'rows.jsonl').write_text(''.join(json.dumps(dict(id=r['id'],page=r['page'],gold=r['gold'],outputs={k:v[i].tolist() for k,v in out.items()}))+'\n' for i,r in enumerate(ev)))
    print(json.dumps(result))
if __name__=='__main__':main()
