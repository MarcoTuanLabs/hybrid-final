"""Retrieve dated evidence from the complete corpus, then rerank retrieved passages."""
import argparse
from collections import Counter,defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import time
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

P=Path(__file__).resolve().parent
STOP=set('what who how many which is are the a an did does was were do in on at of to for and as s with by from that have has be it their its'.split())
def words(s):return [w for w in re.findall(r'[^\W_]+',s.casefold()) if w not in STOP]
def read(f):return json.loads((P/f).read_text())

class BM25:
    def __init__(self,texts):
        self.post=defaultdict(list);self.lengths=[]
        for j,t in enumerate(texts):
            c=Counter(words(t));self.lengths.append(sum(c.values()))
            for term,count in c.items():self.post[term].append((j,count))
        self.lengths=np.array(self.lengths);self.n=len(self.lengths);self.mean=self.lengths.mean()
    def idf(self,t):
        return math.log((1+self.n)/(.5+len(self.post.get(t,[]))))
    def search(self,terms,k):
        scores=defaultdict(float)
        for term,weight in terms.items():
            idf=self.idf(term)*weight
            for i,freq in self.post.get(term,[]):
                scores[i]+=idf*2.2*freq/(freq+1.2*(.25+.75*self.lengths[i]/self.mean))
        return heapq.nsmallest(k,scores,key=lambda i:(-scores[i],i))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);a=ap.parse_args()
    begin=time.monotonic();bank=list(map(json.loads,(P/'bank.jsonl').read_text().splitlines()))
    latest=[row[2]+' '+row[2]+' '+row[6] for row in bank]
    old=[row[2]+' '+row[2]+' '+row[4] for row in bank]
    b=BM25(latest);bo=BM25(old);index_s=time.monotonic()-begin
    with (a.model/'pytorch_model.bin').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8'
    torch.set_num_threads(4);net=AutoModel.from_pretrained(a.model,local_files_only=True).eval();tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    @torch.inference_mode()
    def emb(texts):
        out=[]
        for j in range(0,len(texts),24):
            x=tok(texts[j:j+24],return_tensors='pt',padding=True,truncation=True,max_length=160)
            mask=x['attention_mask'].unsqueeze(-1);out.append(((net(**x).last_hidden_state*mask).sum(1)/mask.sum(1)).numpy())
        return np.vstack(out)
    out={};times={}
    for q in read('questions.json'):
        t=time.monotonic();terms=dict.fromkeys(words(q['question']),1.)
        recent=b.search(terms,32);older=bo.search(terms,32)
        if not recent:recent=[0]
        additions=Counter()
        for i in recent[:3]:
            for w in set(words(latest[i]))-set(terms):additions[w]+=b.idf(w)
        expansion={w:.25 for w,_ in sorted(additions.items(),key=lambda item:(-item[1],item[0]))[:5]}
        fresh=b.search(terms|expansion,32)
        pool=list(dict.fromkeys([*recent,*older,*fresh]))
        vec=emb([q['question']])[0];dense=(emb([bank[i][2]+' — '+bank[i][6] for i in pool])@vec)
        dr=sorted(range(len(pool)),key=lambda j:(-dense[j],pool[j]))
        sr=lambda rank:{v:1/(12+k) for k,v in enumerate(rank)}
        cur,prior,ex,sem=[sr(x) for x in (recent,older,fresh,[pool[j] for j in dr])]
        direct=sorted(recent,key=lambda i:-(cur[i]+sem[i]))
        union=sorted(pool,key=lambda i:-(cur.get(i,0)+.5*prior.get(i,0)+.5*ex.get(i,0)+sem[i]))
        def render(indices):return [dict(key=bank[i][0],page=bank[i][1],title=bank[i][2],old_date=bank[i][3],old=bank[i][4],date=bank[i][5],current=bank[i][6]) for i in indices[:5]]
        out[q['id']]=dict(bm25=render(recent[:1]),direct=render(direct),expanded=render(union),pool=len(pool),extra=sum(i not in recent for i in pool))
        times[q['id']]=time.monotonic()-t
    (P/'retrieved.json').write_text(json.dumps(out,ensure_ascii=False)+'\n')
    (P/'rank_receipt.json').write_text(json.dumps(dict(index_s=index_s,query_s=times,n=len(bank),t=time.monotonic()-begin))+'\n')
if __name__=='__main__':main()
