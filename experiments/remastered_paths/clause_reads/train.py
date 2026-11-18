"""Fit identical CTC relation decoders on lexical, contextual and recurrent-write features."""
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn

P=Path(__file__).resolve().parent
read=lambda f:json.loads(f.read_text())
MODES={
    'lexical':['lex','pos'],
    'residual':['lex','pos','resid'],
    'residual_value':['lex','pos','resid','write'],
    'residual_read':['lex','pos','resid','read'],
    'raw_alignment':['lex','pos','resid','dot'],
    'transport':['lex','pos','resid','transport'],
}

class Decoder(nn.Module):
    def __init__(self,width,num):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(width,80),nn.LayerNorm(80),nn.GELU(),nn.Dropout(.15))
        self.conv=nn.Conv1d(80,80,kernel_size=3,padding=1)
        self.output=nn.Sequential(nn.LayerNorm(80),nn.GELU(),nn.Linear(80,num))

    def forward(self,x):
        h=self.project(x)
        h=h+self.conv(h.transpose(1,2)).transpose(1,2)
        return self.output(h).log_softmax(-1)


def greedy(arr):
    return tuple(k for k,_ in itertools.groupby(arr.argmax(-1).tolist()) if k)


def ladd(a,b):
    if a==-math.inf:return b
    if b==-math.inf:return a
    return max(a,b)+math.log1p(math.exp(-abs(a-b)))


def beam(logprobs,transition,scale):
    # Sum all CTC alignments of a prefix before pruning; transition scores influence only prefix ranking.
    beams={(): (0.,-math.inf)}
    cache={():0.}
    def bonus(seq):
        if seq not in cache:
            cache[seq]=bonus(seq[:-1])+(float(transition[seq[-2],seq[-1]]) if len(seq)>1 else 0.)
        return cache[seq]
    def update(mapping,prefix,blank,score):
        a,b=mapping.get(prefix,(-math.inf,-math.inf))
        mapping[prefix]=(ladd(a,score),b) if blank else (a,ladd(b,score))
    for probs in logprobs:
        labels=np.argpartition(probs[1:],-5)[-5:]+1
        candidates={}
        for prefix,(eb,en) in beams.items():
            total=ladd(eb,en);update(candidates,prefix,True,total+float(probs[0]))
            for token in labels.tolist():
                lp=float(probs[token]);same=bool(prefix) and prefix[-1]==token
                if same:update(candidates,prefix,False,en+lp)
                if len(prefix)<4:update(candidates,prefix+(token,),False,(eb if same else total)+lp)
        def rank(item):return ladd(*item[1])+scale*bonus(item[0])
        beams=dict(sorted(candidates.items(),key=rank,reverse=True)[:10])
    return max(beams,key=lambda pref:ladd(*beams[pref])+scale*bonus(pref))


def fit(x,y,lens,train,valid,probe,transition):
    rng=np.random.default_rng(4103);torch.manual_seed(4103)
    net=Decoder(x.shape[-1],int(y.max())+1)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.02)
    loss=nn.CTCLoss(blank=0,zero_infinity=True)
    F=torch.from_numpy(x);Y=torch.tensor(y);L=torch.tensor(lens);T=(Y>0).sum(1)
    def score(pred,ids):
        return sum(tuple(y[i,:int(T[i])])==v for i,v in zip(ids,pred,strict=True))
    best=None
    for epoch in range(1,37):
        net.train()
        for sample in np.array_split(rng.permutation(train),math.ceil(len(train)/48)):
            xb=F[sample];yb=Y[sample];lb=L[sample];tb=T[sample]
            optimizer.zero_grad(set_to_none=True);lp=net(xb).transpose(0,1)
            err=loss(lp,yb,lb,tb);err.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
        if epoch in (8,18,36):
            net.eval()
            with torch.inference_mode():v=net(F[valid]).numpy()
            acc=score([greedy(v[j,:lens[i]]) for j,i in enumerate(valid)],valid)
            if best is None or acc>best[0]:best=(acc,epoch,{k:t.detach().clone() for k,t in net.state_dict().items()})
    net.load_state_dict(best[2]);net.eval();ids=np.concatenate([valid,probe])
    with torch.inference_mode():out=net(F[ids]).numpy()
    sets={}
    for mode,scale in [('greedy',None),('beam',0.),('transition',.50)]:
        pred=[greedy(v[:lens[i]]) if scale is None else beam(v[:lens[i]],transition,scale) for i,v in zip(ids,out,strict=True)]
        sets[mode]=(score(pred[:len(valid)],valid),pred)
    winner=max(sets,key=lambda m:sets[m][0])
    return best[1],winner,sets[winner][0],sets[winner][1][len(valid):]


def main():
    reg=read(P/'train_contract.json')
    for f,h in reg['pins'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h,f
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows=read(P.parent/'state_parser/items.json');gt=read(P.parent/'state_parser/targets.json')
    keys=read(P.parent/'remastered/labels.json')
    data=np.load(P/'features.npz');lens=data['lengths'];rels=['END',*sorted(read(P.parent/'typed_reads/schema.json'))]
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train' and set(gt[r['id']][1])<=set(rels)]);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='exposed'])
    y=np.zeros((len(rows),4),dtype=np.int64)
    for i,r in enumerate(rows):
        seq=gt[r['id']][1];y[i,:len(seq)]=[rels.index(label) if label in rels else -1 for label in seq]
    freq=np.full((len(rels),len(rels)),.25)
    for i in train:
        path=y[i][y[i]>0]
        for a,b in zip(path,path[1:]):freq[a,b]+=1
    prior=np.log(freq/freq.sum(1,keepdims=True))-np.log(freq.sum(0,keepdims=True)/freq.sum())
    use=np.arange(data['resid'].shape[1])[None,:]<lens[:,None];f={}
    for name in ['lex','pos','resid','write','read','dot','transport']:
        x=data[name].astype(np.float32)
        cells=x[train][use[train]];scale=np.maximum(cells.std(0),1e-5)
        x=np.clip((x-cells.mean(0))/scale,-6,6);x[~use]=0
        f[name]=x
    start=time.monotonic();stats={}
    for mode,names in MODES.items():
        ep,rule,result,plans=fit(np.concatenate([f[n] for n in names],axis=-1),y,lens,train,val,test,prior.clip(-2,2))
        pred={rows[i]['id']:[rels[t] for t in plan] for i,plan in zip(test,plans,strict=True)}
        score=sum(pred[u]==gt[u][1] for u in keys if keys[u]['structure']=='multi')
        stats[mode]=dict(epoch=ep,decode=rule,validation_exact=result,exposed_exact=score,exposed_id_plans=pred)
        print(mode,ep,rule,result,score,round(time.monotonic()-start,1),flush=True)
    cand=stats['transport'];other=[v for k,v in stats.items() if k!='transport']
    promote=cand['exposed_exact']>=max(v['exposed_exact'] for v in other)+3 and cand['validation_exact']>=max(v['validation_exact'] for v in other)
    (P/'4b_parser_scores.json').write_text(json.dumps(dict(ntrain=len(train),nval=len(val),nexposed=48,stats=stats,promote=promote,wall_s=time.monotonic()-start),indent=2)+'\n')
if __name__=='__main__':main()
