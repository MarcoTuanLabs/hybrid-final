"""Fresh dense + typed-edge retrieval. Frozen ranking is shared by the reader ablations."""
import json,sys,time,hashlib,re
from pathlib import Path
import os, gzip
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
p=Path(__file__).resolve().parent
cfg=json.loads((p/'contract.json').read_text())
for file,sha in cfg['pins'].items():assert hashlib.sha256((p/file).read_bytes()).hexdigest()==sha,file
weights=Path(os.environ['HYBRID_CONTRIEVER_WEIGHTS'])
with (weights/'pytorch_model.bin').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==cfg['contriever_sha']
torch.set_num_threads(4)
model=AutoModel.from_pretrained(weights,local_files_only=True).eval()
tokenizer=AutoTokenizer.from_pretrained(weights,local_files_only=True)
@torch.inference_mode()
def embed(texts):
    ids=tokenizer(texts,padding=True,truncation=True,return_tensors='pt');mask=ids['attention_mask'].unsqueeze(-1)
    return ((model(**ids).last_hidden_state*mask).sum(1)/mask.sum(1)).numpy(),int(ids['input_ids'].numel())
norm=lambda s:' '+' '.join(re.findall(r'\w+',s.lower()))+' '
def propagate(x,src,dst):
    z=np.zeros_like(x);np.maximum.at(z,dst,x[src]);return z
out={};build=[]
for bi,b in enumerate(json.loads(gzip.decompress((p/'banks.json.gz').read_bytes()))):
    docs=b['documents'];begin=time.perf_counter()
    chunks=[embed([d['text'] for d in docs[i:i+16]]) for i in range(0,len(docs),16)]
    matrix=np.concatenate([x[0] for x in chunks]);byname={}
    for i,d in enumerate(docs):byname.setdefault(d['sid'],[]).append(i)
    edges=[(i,j) for i,d in enumerate(docs) for j in byname.get(d['oid'],[]) if i!=j]
    pairs=set(edges);src,dst=np.array(edges,dtype=int).reshape(-1,2).T
    build.append(dict(index_s=time.perf_counter()-begin,index_encoder_padded_positions=sum(x[1] for x in chunks)))
    for item in b['queries']:
        begin=time.perf_counter();q=item['question'];v,n=embed([q]);dense=(v@matrix.T)[0]
        order=np.argsort(-dense,kind='stable');rank=np.argsort(order)+1
        seed=1/rank+np.array([norm(d['subject']) in norm(q) for d in docs])
        fwd=propagate(seed,src,dst)
        score=seed+.8*fwd+.64*propagate(fwd,src,dst)+.4*propagate(seed,dst,src)
        best=np.argsort(-score,kind='stable')[:4].tolist();comp={best[0]}
        for _ in range(3):
            comp|={i for i in best if any((i,j) in pairs or (j,i) in pairs for j in comp)}
        out[item['id']]=dict(adaptive=sorted(best[:len(comp)]),dense4=sorted(order[:4].tolist()),rank_s=time.perf_counter()-begin,encoder_positions=n)
(p/'ranking.json').write_text(json.dumps(out)+'\n');(p/'ranking_cost.json').write_text(json.dumps(build)+'\n');print(build)
