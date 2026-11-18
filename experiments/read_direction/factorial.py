"""Replay the fixed memory write with independently exchanged raw read directions and lengths."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
import numpy as np

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(path.read_text())

def replay(trial,cache,suffix,plans,mode,baseline=None,edited=None):
    standard=trial.kernel;linear=sum(layer.is_linear for layer in trial.model.model.layers)
    ordinal=0;captured=[]
    def wrap(*args,**kwargs):
        nonlocal ordinal
        step,layer=divmod(ordinal,linear);ordinal+=1
        result,state=standard(*args,**kwargs)
        if layer!=trial.ordinal:return result,state
        head=result[:,:,28:30,:]
        if mode=='capture':
            captured.append(np.asarray(head.astype(mx.float32)).copy())
            return result,state
        x=baseline[step].astype('float64');y=edited[step].astype('float64')
        bx=np.linalg.norm(x,axis=-1,keepdims=True);ey=np.linalg.norm(y,axis=-1,keepdims=True)
        assert np.all(bx>0) and np.all(ey>0)
        if mode=='direction':new=y*(bx/ey)
        elif mode=='magnitude':new=x*(ey/bx)
        elif mode=='reset':new=x
        else:raise ValueError(mode)
        replaced=mx.concatenate((result[:,:,:28],mx.array(new,dtype=head.dtype),result[:,:,30:]),axis=2)
        return replaced,state
    trial.kernel=wrap
    try:logits,_=trial.steps(cache,suffix,plans)
    finally:
        trial.kernel=standard
        qwen.gated_delta_update=standard
    assert ordinal==len(suffix)*linear
    return logits,captured

def geometry(base,edit):
    x=np.concatenate(base).astype('float64')[:,0];y=np.concatenate(edit).astype('float64')[:,0]
    bx=np.linalg.norm(x,axis=-1,keepdims=True);by=np.linalg.norm(y,axis=-1,keepdims=True)
    u=x/bx;v=y/by;n=x.shape[-1]
    rx=bx/np.sqrt(bx**2/n+1e-6);ry=by/np.sqrt(by**2/n+1e-6)
    turn=(rx+ry)/2*(v-u);radial=(ry-rx)*(u+v)/2
    direct=ry*v-rx*u
    assert np.allclose(turn+radial,direct,atol=1.e-11)
    assert np.allclose(np.sum(turn*radial,axis=-1),0.,atol=1.e-11)
    angular_energy=float(np.sum(turn**2));radial_energy=float(np.sum(radial**2))
    angles=np.rad2deg(np.arccos(np.clip(np.sum(u*v,axis=-1),-1,1)))
    return dict(mean_raw_angle_degrees=float(angles.mean()),max_raw_angle_degrees=float(angles.max()),
                angular_energy=angular_energy,radial_energy=radial_energy,
                total_energy=float(np.sum(direct**2)),min_raw_norm=float(min(bx.min(),by.min())))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=('dev','test'));ap.add_argument('model',type=Path);ap.add_argument('--public-repo',type=Path,default=HERE.parents[1]);a=ap.parse_args()
    repo=a.public_repo.resolve();module=repo/'experiments/quantized_write'
    sys.path.insert(0,str(module))
    from run import Trial,make_example
    receipt=read(repo/'experiments/read_projection/release.json')
    for name,sha in receipt['model_files'].items():
        with (a.model/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha,name
    src=repo/'evidence/quantized_9b';cfg=read(src/'contract.json')
    phase=a.phase
    if phase=='test':
        assert read(HERE/'dev_summary.json')['gate']
        lock=read(HERE/'lock.json')
        for name,digest in lock.items(): assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    seeds=cfg['seeds'] if phase=='dev' else list(range(96374120,96374184))
    arc=list(map(json.loads,gzip.decompress((src/'rows.jsonl.gz').read_bytes()).decode().splitlines()))
    archived={(v['seed'],v['surface'],v['nuisance'],v['written'],v['mode']):v for v in arc}
    mx.set_cache_limit(64*1024**2);trial=Trial(a.model,cfg)
    path=HERE/(phase+'_rows.jsonl');done=set()
    if path.exists():done={(v['seed'],v['surface'],v['nuisance']) for v in map(json.loads,path.read_text().splitlines())}
    begin=time.monotonic()
    with path.open('a') as f:
        for idx,seed in enumerate(seeds):
            for surface in ['table','prose']:
                for nuisance in (False,True):
                    key=(seed,surface,nuisance)
                    if key in done:continue
                    t=time.monotonic();ex=make_example(*key)
                    left,tail=trial.encode(ex,'a');right,tail2=trial.encode(ex,'b');assert tail==tail2
                    base=trial.prefill(left);donor=trial.prefill(right);edit=trial.write(base,donor)
                    lb,le,slots,errs=trial.geometry(base,edit,tail)
                    plans=[trial.solve(g,'ref16') for g in slots];white=[trial.solve(g,'white') for g in slots]
                    l0,x=replay(trial,base,tail,plans,'capture');l1,y=replay(trial,edit,tail,plans,'capture')
                    native0,xn=replay(trial,base,tail,None,'capture')
                    native1,yn=replay(trial,edit,tail,None,'capture')
                    assert np.array_equal(native0,lb) and np.array_equal(native1,le)
                    lwhite,_=trial.steps(edit,tail,white)
                    scores={'unwritten':l0,'full':l1,'native_written':le,'native_unwritten':lb,'white_written':lwhite}
                    for mode in ['direction','magnitude','reset']:
                        scores[mode]=replay(trial,edit,tail,plans,mode,x,y)[0]
                    donor_id,recipient_id=[trial.tok.encode(ex[k],add_special_tokens=False)[0] for k in ['value_b','value_a']]
                    if phase=='dev':
                        for name,written,m in [('full',True,'ref16'),('unwritten',False,'ref16'),('white_written',True,'white'),('native_written',True,'native'),('native_unwritten',False,'native')]:
                            v=archived[(*key,written,m)];logits=scores[name]
                            assert int(logits.argmax())==v['top'] and abs(float(logits[donor_id]-logits[recipient_id])-v['margin'])<1e-5,(key,name)
                    reset_max=float(np.max(abs(scores['reset']-l0)));assert reset_max<=1e-5,(key,reset_max)
                    outputs={m:dict(top=int(v.argmax()),margin=float(v[donor_id]-v[recipient_id])) for m,v in scores.items()}
                    row=dict(seed=seed,surface=surface,nuisance=nuisance,recipient_id=recipient_id,donor_id=donor_id,outputs=outputs,
                             reset_max_logit_error=reset_max,read_geometry=geometry(x,y),native_geometry=geometry(xn,yn),seconds=time.monotonic()-t)
                    f.write(json.dumps(row)+'\n');f.flush();mx.clear_cache()
            print(idx+1,len(seeds),round(time.monotonic()-begin,1),flush=True)
            if time.monotonic()-begin>(900 if phase=='dev' else 2100):raise TimeoutError('Fixed experiment budget')
if __name__=='__main__':main()
