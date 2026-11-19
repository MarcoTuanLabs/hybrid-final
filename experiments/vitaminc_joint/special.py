"""Evaluate the published VitaminC specialists with the authors' passage/claim order."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from pairs import HERE,build,public_split
from specialist import MODELS,LABELS
import torch
from transformers import AlbertTokenizer,AlbertForSequenceClassification

WEIGHTS={"albert-base-vitaminc-mnli":"c48e88efef4e2ae6bf8229558fc50d1dd84681b52d7d78889e88d2f61d89f0b3",
         "albert-xlarge-vitaminc":"9bdb51b39fa05791282ff19137d6d5888d22c7d6c2629ef72e28641b717c14a7"}

def main():
    p=argparse.ArgumentParser()
    for n in ('data','model','output'):p.add_argument(n,type=Path)
    a=p.parse_args();m=a.model.name;parts=build(a.data)
    assert public_split(parts)==json.loads((HERE/'split.json').read_text())
    with (a.model/'model.safetensors').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==WEIGHTS[m]
    torch.set_num_threads(4)
    tok=AlbertTokenizer.from_pretrained(a.model,local_files_only=True)
    net=AlbertForSequenceClassification.from_pretrained(a.model,local_files_only=True).eval().to('mps')
    assert [net.config.id2label[i] for i in range(3)]==LABELS
    a.output.mkdir(parents=True,exist_ok=True)
    for part,items in parts.items():
        start=time.monotonic();rows=[]
        with torch.inference_mode():
            for i in range(0,len(items),4):
                subset=items[i:i+4]
                passages=[s for r in subset for s in r['passages']]
                claims=[r['claim'] for r in subset for _ in range(2)]
                tokens=tok(passages,claims,padding=True,truncation=True,max_length=256,return_tensors='pt').to('mps')
                pred=net(**tokens).logits.cpu().reshape(len(subset),2,3)
                for r,y in zip(subset,pred):rows.append(dict(id=r['id'],gold=r['y'],logits=y.tolist()))
                if time.monotonic()-start>900:raise TimeoutError(m)
        (a.output/(m+'-'+part+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(m,MODELS[m],part,len(rows),round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':main()
