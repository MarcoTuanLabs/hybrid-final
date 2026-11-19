"""Fit readouts on calibration pages and score complete, separate evaluation pages."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def ridge(x,y,center=True):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    a=x.mean(0) if center else np.zeros(x.shape[1]);b=y.mean(0) if center else np.zeros(y.shape[1])
    u=x-a;gram=u@u.T
    gram.flat[::len(gram)+1]+=0.05*np.trace(gram)/len(gram)+1e-9
    w=u.T@np.linalg.solve(gram,y-b)
    return lambda v:(v-a)@w+b


def prob(z):
    z=z-z.max(-1,keepdims=True);v=np.exp(z)
    return v/v.sum(-1,keepdims=True)


def load_part(items,cache):
    keys=["h","lens","full","gold","tokens"];out={k:[] for k in keys}
    for row in items:
        with np.load(cache/(row["id"]+".npz")) as f:
            assert np.isfinite(f['h']).all() and f['h'].shape==(2,2,4096)
            for k in keys:out[k].append(f[k])
    return {k:np.concatenate(v).astype(np.float64 if k not in ("gold","tokens") else np.int32) for k,v in out.items()}


def flat(a):
    return a.reshape(-1,a.shape[-1])


def methods(tr,ev):
    oh=np.eye(3)[tr['gold']]
    old=lambda x:np.flip(x,axis=1)
    ft=tr['full']-tr['full'].mean(-1,keepdims=True)
    fe=ev['full']-ev['full'].mean(-1,keepdims=True)
    dt=tr['h']-old(tr['h']);de=ev['h']-old(ev['h'])
    delta=ridge(flat(dt),flat(ft-old(ft)),False)
    absolute=ridge(flat(tr['h']),flat(ft))
    supervised=ridge(flat(tr['h']),flat(oh))
    supervised_delta=ridge(flat(dt),flat(oh-old(oh)),False)
    stale=old(fe)
    # Calibrated odds of seeing the two labels together after a revision.
    joint=np.full((3,3),2.)
    for a,b in tr['gold']:joint[a,b]+=1;joint[b,a]+=1
    joint/=joint.sum();assoc=joint/np.outer(joint.sum(1),joint.sum(0))
    def revision_prior(new):
        return np.exp(new-new.max(-1,keepdims=True))*(prob(stale)@assoc)
    return dict(full=fe,stale=stale,raw_lens=ev['lens'],absolute_teacher=absolute(ev['h']),
                absolute_labels=supervised(ev['h']),paired=stale+delta(de),
                paired_labels=prob(stale)+supervised_delta(de),
                native_delta=stale+ev['lens']-old(ev['lens']),
                old_transition=prob(stale)@(joint/joint.sum(-1,keepdims=True)),
                prior_absolute=revision_prior(absolute(ev['h'])),prior_full=revision_prior(fe),
                full_contrast=2*fe-stale)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("cache",type=Path);ap.add_argument("part",choices=["pilot","confirmation"]);ap.add_argument("output",type=Path);a=ap.parse_args()
    sp=json.loads((HERE/"split.json").read_text());tr=load_part(sp['fit'],a.cache/'fit');ev=load_part(sp[a.part],a.cache/a.part)
    raw=methods(tr,ev);pred={k:v.argmax(-1) for k,v in raw.items()}
    g=ev['gold'];stable=np.repeat((g[:,0]==g[:,1])[:,None],2,axis=1)
    score={k:dict(correct=int((v==g).sum()),changed=int(((v==g)&~stable).sum()),stable=int(((v==g)&stable).sum()),
                  pair_both=int((v==g).all(1).sum()),endpoint_wins_vs_full=int(((v==g)&(pred['full']!=g)).sum()),
                  endpoint_losses_vs_full=int(((v!=g)&(pred['full']==g)).sum())) for k,v in pred.items()}
    s=dict(pages=len(sp[a.part]),claims=len(g),endpoints=g.size,stable=int(stable.sum()),changed=int((~stable).sum()),label_counts=dict(Counter(map(str,g.flat))),methods=score)
    same=("absolute_teacher","absolute_labels","raw_lens","old_transition","prior_absolute")
    s["promote"]=bool(score["paired"]["correct"]>=score["full"]["correct"] and score["paired"]["correct"]>=max(score[k]["correct"] for k in same)+4 and score["paired"]["stable"]>=score["full"]["stable"]-1)
    rows=[]
    for i,r in enumerate(sp[a.part]):
        span=slice(i*2,i*2+2)
        rows.append(dict(id=r['id'],page=r['page'],gold=g[span].tolist(),tokens=ev['tokens'][span].tolist(),outputs={k:v[span].tolist() for k,v in pred.items()},full_logits=ev['full'][span].tolist(),paired_logits=raw['paired'][span].tolist()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    (a.output/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps(s))
if __name__=="__main__":main()
