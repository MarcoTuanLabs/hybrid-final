"""Fit symmetric and antisymmetric components with separate scales."""
import argparse
import json
from pathlib import Path
import numpy as np
from score import HERE,load_part,ridge,flat


def predict(tr,ev):
    y=np.eye(3)[tr['gold']];x=tr['h'];z=ev['h']
    reverse=lambda t:np.flip(t,axis=1)
    even=lambda t:(t+reverse(t))/2
    odd=lambda t:(t-reverse(t))/2
    direct=ridge(flat(x),flat(y))
    center=ridge(even(x)[:,0],even(y)[:,0])
    difference=ridge(flat(odd(x)),flat(odd(y)),False)
    single=direct(z);middle=center(even(z));delta=difference(odd(z))
    shared=even(single)+odd(single)
    assert np.max(np.abs(shared-single))<1e-9
    return dict(direct=single,paired_components=middle+delta,shared_mean=even(single)+delta,
                shared_difference=middle+odd(single),mean_only=middle)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("cache",type=Path);ap.add_argument("part",choices=("pilot","confirmation"));ap.add_argument("output",type=Path);a=ap.parse_args()
    sp=json.loads((HERE/"split.json").read_text())
    tr=load_part(sp["fit"],a.cache/"fit");ev=load_part(sp[a.part],a.cache/a.part)
    probs=predict(tr,ev);g=ev['gold'];stable=np.broadcast_to(g[:,0,None]==g[:,1,None],g.shape)
    rows=[dict(id=r['id'],page=r['page'],gold=g[i*2:i*2+2].tolist(),outputs={k:v[i*2:i*2+2].argmax(-1).tolist() for k,v in probs.items()}) for i,r in enumerate(sp[a.part])]
    ref=[json.loads(v) for v in (a.output/'rows.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==[r['id'] for r in ref]
    methods={k:v.argmax(-1) for k,v in probs.items()}
    for label in ('full','paired_labels','old_transition','absolute_labels','full_contrast'):
        methods[label]=np.concatenate([r['outputs'][label] for r in ref])
    for name in ('albert-base-vitaminc-mnli','albert-xlarge-vitaminc'):
        records=[json.loads(v) for v in (a.output/(name+'-'+a.part+'.jsonl')).read_text().splitlines()]
        assert [r['id'] for r in rows]==[r['id'] for r in records] and [r['gold'] for r in rows]==[r['gold'] for r in records]
        methods[name]=np.concatenate([r['answers'] for r in records])
    assert np.array_equal(methods['direct'],methods['absolute_labels'])
    result={k:dict(correct=int((v==g).sum()),changed=int(((v==g)&~stable).sum()),stable=int(((v==g)&stable).sum()),both=int((v==g).all(1).sum())) for k,v in methods.items()}
    primary=result['paired_components']
    advance=primary['correct']>=max(result[k]['correct'] for k in ('direct','full','paired_labels','old_transition','full_contrast'))+4 and primary['correct']>=max(result[k]['correct'] for k in ('albert-base-vitaminc-mnli','albert-xlarge-vitaminc')) and primary['stable']>=result['direct']['stable']-1
    summary=dict(pages=len(rows),endpoints=g.size,stable=int(stable.sum()),methods=result,promote=advance)
    (a.output/'component_rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (a.output/'component_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=="__main__":main()
