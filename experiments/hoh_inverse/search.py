"""Search with a hypothetical passage, then compare generated questions in the same embedding space."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'hoh_updates'
sys.path.insert(0,str(SOURCE))
from rank import BM25,words

def read(p):
    return json.loads(p.read_text())

class Encoder:
    def __init__(self,model):
        with (model/'pytorch_model.bin').open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()=='08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8'
        torch.set_num_threads(4)
        self.net=AutoModel.from_pretrained(model,local_files_only=True).eval()
        self.tok=AutoTokenizer.from_pretrained(model,local_files_only=True)

    @torch.inference_mode()
    def encode(self,texts):
        vec=[]
        for start in range(0,len(texts),24):
            ids=self.tok(texts[start:start+24],padding=True,truncation=True,max_length=160,return_tensors='pt')
            mask=ids.attention_mask.unsqueeze(-1)
            x=(self.net(**ids).last_hidden_state*mask).sum(1)/mask.sum(1)
            vec.append(x.numpy())
        return np.vstack(vec)

def reciprocal(order):
    return {k:1/(12+pos) for pos,k in enumerate(order)}

def retrieve(enc):
    start=time.monotonic()
    bank=list(map(json.loads,(SOURCE/'bank.jsonl').read_text().splitlines()))
    index=BM25([row[2]+' '+row[2]+' '+row[6] for row in bank])
    keys={row[0]:i for i,row in enumerate(bank)}
    old_ranks=read(SOURCE/'retrieved.json');rewrites=read(HERE/'rewrites.json');results={}
    for q in read(SOURCE/'questions.json'):
        uid=q['id'];prior=[keys[r['key']] for r in old_ranks[uid]['direct']]
        other=[keys[r['key']] for r in old_ranks[uid]['expanded']]
        original=dict.fromkeys(words(q['question']),1.)
        extra=dict.fromkeys(words(rewrites[uid]['hypothetical']),.65)
        direct=index.search(original,32);new=index.search(extra|original,32)
        pool=list(dict.fromkeys(prior+other+direct+new))
        vec=enc.encode([q['question'],rewrites[uid]['hypothetical']])
        doc=enc.encode([bank[i][2]+' — '+bank[i][6] for i in pool])
        qdots=doc@vec.T
        dense=sorted(pool,key=dict(zip(pool,qdots[:,0],strict=True)).get,reverse=True)
        hypothetical=sorted(pool,key=dict(zip(pool,qdots[:,1],strict=True)).get,reverse=True)
        ranks=[reciprocal(order) for order in (prior,direct,new,dense,hypothetical)]
        ordered=sorted(pool,key=lambda i:-sum(r.get(i,0) for r in ranks))
        selected=ordered[:4]
        novel=[i for i in ordered if i not in set(prior+other+selected)]
        selected+=novel[:2]
        selected+=[i for i in ordered if i not in selected][:6-len(selected)]
        assert len(set(selected))==6
        results[uid]=[dict(key=bank[i][0],page=bank[i][1],title=bank[i][2],old=bank[i][4],
                           current=bank[i][6],old_date=bank[i][3],date=bank[i][5]) for i in selected]
    (HERE/'pool.json').write_text(json.dumps(results,ensure_ascii=False)+'\n')
    (HERE/'search_time.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,corpus=len(bank)))+'\n')

def card_scores(enc):
    start=time.monotonic()
    cards={r['key']:r for r in map(json.loads,(HERE/'cards.jsonl').read_text().splitlines())}
    pool=read(HERE/'pool.json');scores={}
    for q in read(SOURCE/'questions.json'):
        keys=[r['key'] for r in pool[q['id']]]
        texts=[cards[k][mode]['question'] or cards[k][mode]['raw'] or '(empty)' for k in keys for mode in ('current','paired')]
        vec=enc.encode([q['question'],*texts])
        sims=vec[1:]@vec[0];scores[q['id']]={k:dict(current=float(sims[2*i]),paired=float(sims[2*i+1])) for i,k in enumerate(keys)}
    (HERE/'card_similarity.json').write_text(json.dumps(scores)+'\n')
    (HERE/'card_embedding_time.json').write_text(json.dumps(dict(seconds=time.monotonic()-start))+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['search','cards']);p.add_argument('model',type=Path);a=p.parse_args()
    enc=Encoder(a.model)
    (retrieve if a.phase=='search' else card_scores)(enc)
