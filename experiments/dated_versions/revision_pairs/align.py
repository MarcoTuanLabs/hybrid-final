"""Align two archived article editions and retrieve changed fields before generation."""
import argparse
from collections import Counter
import difflib
import json
import math
from pathlib import Path
import re
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from transformers import AutoModel,AutoTokenizer

P=Path(__file__).resolve().parent
S=P.parent/'asof_revision'


def tokens(text):
    return re.findall(r'[\w]+',text.casefold())


def parts(ed):
    out=[]
    for row in ed['rows']:
        cells=[' '.join(c.split()) for c in row['cells']]
        if len(cells)>=2:
            out.append(('field',row['section'],cells[0],' | '.join(cells[1:])))
        elif cells:
            out.append(('heading','','',cells[0]))
    for para in ed['paragraphs']:
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])',para):
            if len(sentence)>25:out.append(('prose','','',' '.join(sentence.split()[:60])))
    return list(dict.fromkeys(out))


def align(old,new):
    matches=[]
    for kind in ('field','heading','prose'):
        left=[r for r in old if r[0]==kind];right=[r for r in new if r[0]==kind]
        if not left or not right:continue
        tok=[[(set(tokens(r[3])),set(tokens(r[2]))) for r in side] for side in (left,right)]
        sim=np.zeros((len(left),len(right)))
        for i,a in enumerate(left):
            for j,b in enumerate(right):
                aw,ak=tok[0][i];bw,bk=tok[1][j]
                common=len(aw&bw)/max(1,len(aw|bw))
                name=len(ak&bk)/max(1,len(ak|bk))
                same=bool(ak and ak==bk)
                value_equal=(a[3].casefold()==b[3].casefold())
                sim[i,j]=3*same+name+common+.10*(a[1]==b[1])+.15*value_equal
        l,r=linear_sum_assignment(-sim)
        for i,j in zip(l,r,strict=True):
            a,b=left[i],right[j]
            if a[3].casefold()==b[3].casefold():continue
            if sim[i,j]<(1.0 if kind=='field' else .35):continue
            matches.append(dict(kind=kind,key=' | '.join(v for v in (a[1],a[2]) if v),old=a[3],new=b[3],match=float(sim[i,j])))
    return matches


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model');args=ap.parse_args()
    torch.set_num_threads(4);model=AutoModel.from_pretrained(args.model,local_files_only=True).eval()
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    @torch.inference_mode()
    def embeddings(texts):
        batch=tok(texts,truncation=True,max_length=160,padding=True,return_tensors='pt')
        mask=batch['attention_mask'].unsqueeze(-1)
        return ((model(**batch).last_hidden_state*mask).sum(1)/mask.sum(1)).numpy()
    start=time.monotonic();items=json.loads((S/'inputs.json').read_text());out=[]
    for row in items:
        q=row['question'];page=row['page'];docs=align(*map(parts,row['editions']))
        if docs:
            query_no_page=re.sub(re.escape(page),'',q,flags=re.I)
            query=embeddings([query_no_page,q])
            doc_text=[d['key']+' | '+d['old']+' / '+d['new'] for d in docs]
            v=np.concatenate([embeddings(doc_text[j:j+24]) for j in range(0,len(docs),24)])
            keyed=np.concatenate([embeddings([d['key'] or t for d,t in zip(docs[j:j+24],doc_text[j:j+24],strict=True)]) for j in range(0,len(docs),24)])
            dense=(v@query[1])+.5*(keyed@query[0])
            stop=set('who what is the of a an s how many did does do at in which for was current are'.split())
            terms=set(tokens(query_no_page))-stop
            counts=[Counter(tokens(t)) for t in doc_text]
            df={t:sum(t in c for c in counts) for t in terms}
            lexical=[sum(math.log(1+len(docs)/(1+df[t]))*(int(t in c)+2*int(t in tokens(d['key']))) for t in terms) for d,c in zip(docs,counts,strict=True)]
            def ranks(scores):
                return {i:n for n,i in enumerate(sorted(range(len(scores)),key=lambda j:(-scores[j],j)))}
            dr,lr=map(ranks,(dense,lexical))
            docs=[docs[i] for i in sorted(range(len(docs)),key=lambda j:-(1/(6+dr[j])+1/(6+lr[j])))[:5]]
        out.append(dict(id=row['id'],pairs=docs))
    (P/'alignment.json').write_text(json.dumps(out,ensure_ascii=False)+'\n')
    (P/'alignment_receipt.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,empty=sum(not r['pairs'] for r in out)))+'\n')
if __name__=='__main__':main()
