"""Retrieve firsthand statements and extract a question-conditioned answer span from each."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from search import BM25, words, reciprocal

def unique(items):
    return list(dict.fromkeys(items))

def read(path):
    return json.loads(path.read_text())

def quarters(order,rows,count):
    return [key for part in range(4) for key in [j for j in order if rows[j]['quarter']==part][:count]]

def check(path,expected):
    with path.open('rb') as source:
        assert hashlib.file_digest(source,'sha256').hexdigest()==expected

@torch.inference_mode()
def embed(model,tok,texts,device):
    vec=[]
    for i in range(0,len(texts),20):
        x=tok(texts[i:i+20],padding=True,max_length=160,truncation=True,return_tensors='pt').to(device)
        mask=x.attention_mask.unsqueeze(-1)
        v=(model(**x).last_hidden_state*mask).sum(1)/mask.sum(1)
        vec.append(v.cpu().numpy())
    return np.vstack(vec)

@torch.inference_mode()
def spans(model,tok,question,docs,device):
    result=[]
    for i in range(0,len(docs),16):
        texts=docs[i:i+16]
        x=tok([question]*len(texts),texts,padding=True,return_offsets_mapping=True,max_length=384,truncation='only_second',return_tensors='pt')
        out=model(input_ids=x.input_ids.to(device),attention_mask=x.attention_mask.to(device))
        starts=out.start_logits.cpu().numpy();ends=out.end_logits.cpu().numpy()
        for j,s in enumerate(texts):
            valid=np.array([n for n,v in enumerate(x.sequence_ids(j)) if v==1 and x.offset_mapping[j,n,1]>x.offset_mapping[j,n,0]])
            assert len(valid)>0
            lo=valid[:,None];hi=valid[None,:]
            scores=np.where((lo<=hi)&(hi<lo+24),starts[j,lo]+ends[j,hi],-np.inf)
            u,v=np.unravel_index(np.argmax(scores),scores.shape)
            l=int(x.offset_mapping[j,valid[u],0]);r=int(x.offset_mapping[j,valid[v],1])
            result.append(dict(span=s[l:r],start=l,end=r,margin=float(scores[u,v]-starts[j,0]-ends[j,0])))
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('data',type=Path);ap.add_argument('dense',type=Path);ap.add_argument('span',type=Path);a=ap.parse_args()
    check(a.dense/'pytorch_model.bin','08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8')
    check(a.span/'model.safetensors','ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7')
    device='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(3)
    dense=AutoModel.from_pretrained(a.dense,local_files_only=True).eval().to(device);dt=AutoTokenizer.from_pretrained(a.dense,local_files_only=True)
    reader=AutoModelForQuestionAnswering.from_pretrained(a.span,local_files_only=True).eval().to(device);rt=AutoTokenizer.from_pretrained(a.span,local_files_only=True)
    qs=read(a.data/'questions.json');hs=read(a.data/'histories.json');forms=read(HERE/'formulations.json');prev=read(a.data/'evidence.json')
    dest=HERE/'retrieval.jsonl';seen={v['id'] for v in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    start=time.monotonic()
    with dest.open('a') as f:
        for q in qs:
            uid=q['id']
            if uid in seen:continue
            t=time.monotonic();h=hs[uid];rows={}
            for j,sess in enumerate(h):
                for k,turn in enumerate(sess['turns']):
                    if turn['speaker']=='user' and turn['text'].strip():
                        key=str(sess['session'])+'/'+str(k)
                        rows[key]=dict(session=sess['session'],offset=k,date=sess['date'],text=turn['text'],quarter=min(3,4*j//len(h)))
            keys=list(rows);index=BM25([rows[k]['text'] for k in keys])
            def search(query):
                return [keys[j] for j in index.search(dict.fromkeys(words(query),1.),len(keys))]
            ori=search(q['question']);ext=search(forms[uid]['terms']+' '+forms[uid]['question'])
            existing=unique(str(r['session'])+'/'+str(r['offset']+j) for r in prev[uid]['wide'] for j,v in enumerate(r['turns']) if v['speaker']=='user')
            pool=unique(ori[:28]+ext[:36]+quarters(ori,rows,3)+quarters(ext,rows,4)+existing)
            assert pool and all(k in rows for k in pool)
            vec=embed(dense,dt,[q['question'],forms[uid]['terms']]+[rows[k]['text'] for k in pool],device)
            sim=vec[2:]@vec[:2].T
            dr=[[pool[j] for j in np.argsort(-sim[:,i],kind='stable')] for i in range(2)]
            ranks=[reciprocal(order) for order in [ori,ext,*dr]]
            order=sorted(pool,key=lambda k:-sum(m.get(k,0) for m in ranks))
            shortlist=unique(order[:36]+quarters(order,rows,5))
            extracts=spans(reader,rt,forms[uid]['question'],[rows[k]['text'] for k in shortlist],device)
            details={k:rows[k]|v for k,v in zip(shortlist,extracts,strict=True)}
            ranked=sorted(shortlist,key=lambda k:-details[k]['margin'])
            selected=unique(ranked[:12]+quarters(ranked,rows,2))
            current=[k for k in keys if rows[k]['session']==h[-1]['session']]
            record=dict(id=uid,statements=len(rows),pool=len(pool),device=device,form=forms[uid]['question'],rank=shortlist,span_rank=ranked,support=selected,docs=details,current=[rows[k] for k in unique([current[0],current[-1]])],seconds=time.monotonic()-t)
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();print(len(seen)+1,round(time.monotonic()-start,1),flush=True);seen.add(uid)
            if time.monotonic()-start>900:raise TimeoutError('Retrieval budget')
if __name__=='__main__':main()
