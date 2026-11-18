"""Fit the shared read and matrix directions using eligible training documents only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import mlx.core as mx
from documents import HERE,MARK,setup,message,document,encoding,build_cache
from cache_service import capture

def leading(cov):
    _,vec=np.linalg.eigh(cov);return vec[:,:,:,-4:].astype(np.float16)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('public',type=Path);a=ap.parse_args()
    trial,parts=setup(a.public,a.model)
    from codec import observability
    begin=time.perf_counter();read_cov=delta_cov=None;fit_ids=[]
    for v in parts['fit']:
        texts=[message(v['source'],v[field]) for field in ['old','new']]
        old,new=[build_cache(trial,document(trial,txt)) for txt in texts]
        tail=encoding(trial,MARK)[:4];assert len(tail)==4
        trace=capture(trial,new,tail);reads=[];matrices=[]
        for j,i in enumerate(i for i,l in enumerate(trial.model.model.layers) if l.is_linear):
            d=np.asarray((new[i][1]-old[i][1])[0].astype(mx.float32))
            _,w=observability([t[j] for t in trace],len(d))
            w=w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-12)
            reads.append(w@w.swapaxes(1,2));matrices.append(d.swapaxes(1,2)@d)
        if read_cov is None:read_cov=np.stack(reads);delta_cov=np.stack(matrices)
        else:read_cov+=np.stack(reads);delta_cov+=np.stack(matrices)
        fit_ids.append([v['source'],v['id']]);mx.clear_cache()
        if time.perf_counter()-begin>900:raise TimeoutError('Fitting budget')
    generic=leading(read_cov);svd=leading(delta_cov)
    rng=np.random.default_rng(16877);random=np.linalg.qr(rng.standard_normal(generic.shape,dtype=np.float32),mode='reduced')[0].astype(np.float16)
    np.savez(HERE/'bases.npz',read=generic,svd=svd,random=random)
    (HERE/'fit.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,ids=fit_ids,shape=generic.shape,bytes_per_basis=generic.nbytes,positions='four known template-boundary tokens per donor plus the old and new fact prefills',max_orthogonality_error=float(np.max(np.abs(generic.astype('float32').swapaxes(-1,-2)@generic.astype('float32')-np.eye(4))))),indent=2)+'\n')
if __name__=='__main__':main()
