"""Make a page-disjoint natural revision split and collect the fixed specialist's verdict logits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vitaminc_joint'))
import pairs
from pairs import digest
from data import LABELS
TEST_SHA="7ad1808dbc30c62e0a1427a53022d0dfaff668a1fde3c4b612a2d266edd753ad"
WEIGHTS="9bdb51b39fa05791282ff19137d6d5888d22c7d6c2629ef72e28641b717c14a7"
COUNTS=[((2,2),2),((1,1),2),((0,0),24),((1,2),12),((0,2),92),((0,1),92)]


def select(data):
    pairs.PINS['test']=TEST_SHA
    old=pairs.known()|{r['page'] for v in json.loads((HERE.parent/'vitaminc_joint/split.json').read_text()).values() for r in v}
    train=set()
    with (data/'train.jsonl').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==pairs.PINS['train']
    for line in (data/'train.jsonl').open():
        train.add(digest('page',json.loads(line)['page']))
    pool=pairs.corpus(data/'test.jsonl')
    items=pairs.draw(pool,COUNTS,old|train)
    assert len(items)==len({r['page'] for r in items})==224
    return items,dict(pages=len(items),previous_study_pages=len(old),excluded_train_pages=len(train),source_sha256=TEST_SHA,
                      items=[dict(id=r['id'],page=r['page']) for r in items])


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['split','run']);p.add_argument('data',type=Path)
    p.add_argument('--model',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    items,s=select(a.data)
    if a.action=='split':
        (HERE/'split.json').write_text(json.dumps(s,indent=2)+'\n');return
    assert s==json.loads((HERE/'split.json').read_text())
    assert a.model.name=='albert-xlarge-vitaminc'
    with (a.model/'model.safetensors').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==WEIGHTS
    import torch
    from transformers import AlbertTokenizer,AlbertForSequenceClassification
    torch.set_num_threads(4)
    model=AlbertForSequenceClassification.from_pretrained(a.model,local_files_only=True).eval().to('mps')
    tok=AlbertTokenizer.from_pretrained(a.model,local_files_only=True)
    assert [model.config.id2label[k] for k in range(3)]==LABELS
    a.output.mkdir(exist_ok=True,parents=True);rows=[];start=time.monotonic()
    with torch.inference_mode():
        for i in range(0,len(items),4):
            batch=items[i:i+4];texts=[x for row in batch for x in row['passages']];claims=[row['claim'] for row in batch for _ in range(2)]
            tokens=tok(texts,claims,padding=True,truncation=True,max_length=256,return_tensors='pt').to('mps')
            logits=model(**tokens).logits.cpu().reshape(len(batch),2,3)
            for r,y in zip(batch,logits):
                rows.append(dict(id=r['id'],page=r['page'],gold=r['y'],logits=y.tolist()))
            if time.monotonic()-start>900:raise TimeoutError("transfer")
    (a.output/'logits.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(len(rows),'pages',round(time.monotonic()-start,2),'collection seconds',flush=True)
if __name__=='__main__':main()
