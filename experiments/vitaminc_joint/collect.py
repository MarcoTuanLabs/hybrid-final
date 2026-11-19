"""Read both labeled passages with frozen Qwen weights; collect an explicit revision-prompt control."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
from mlx_lm import load
import numpy as np
from pairs import HERE,build,public_split
from run import tokens,read,INSTRUCTION


def main():
    p=argparse.ArgumentParser()
    for n in ('data','model','cache','pins'):p.add_argument(n,type=Path)
    p.add_argument('part',choices=('fit','pilot'));a=p.parse_args();parts=build(a.data)
    assert public_split(parts)==json.loads((HERE/'split.json').read_text())
    for n,d in json.loads(a.pins.read_text())['model_files'].items():
        with (a.model/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==d
    q,t=load(str(a.model));mx.set_cache_limit(96*1024**2)
    ids=[t.encode(c,add_special_tokens=False) for c in 'ABC'];assert all(len(v)==1 for v in ids);labels=mx.array([v[0] for v in ids])
    output=a.cache/a.part;output.mkdir(exist_ok=True,parents=True);start=time.monotonic()
    for i,r in enumerate(parts[a.part]):
        dest=output/(r['id']+'.npz')
        if dest.exists():continue
        hs=[];full=[];scores=[];lengths=[]
        for j,passage in enumerate(r['passages']):
            prompt=tokens(t,passage,r['claim'])
            h,_,y=read(q,prompt,labels,i==0 and j==0);hs.append(h);full.append(y);lengths.append(len(prompt))
            if a.part=='pilot':
                msg=f"{INSTRUCTION}\nThe first passage is superseded. Judge the claim using the CURRENT passage.\n\nSuperseded: {r['passages'][1-j]}\n\nCURRENT passage: {passage}\n\nClaim: {r['claim']}\nVerdict:"
                tok=t.apply_chat_template([{'role':'user','content':msg}],add_generation_prompt=True,enable_thinking=False,tokenize=True)
                score=np.array(q(mx.array([tok]))[0,-1,labels].astype(mx.float32));assert score.shape==(3,);scores.append(score)
        with dest.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f,h=hs,full=full,explicit=scores,gold=r['y'],tokens=lengths)
        dest.with_suffix('.tmp').replace(dest)
        mx.clear_cache()
        if (i+1)%32==0:print(a.part,i+1,round(time.monotonic()-start,2),flush=True)
        if time.monotonic()-start>1200:raise TimeoutError("split budget")
if __name__=='__main__':main()
