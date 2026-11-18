"""Map generated free relation names with the reference G-Walk Contriever threshold."""
import json
import re
from pathlib import Path
import torch
from transformers import AutoModel,AutoTokenizer

P=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
reg=read(P.parent/'remastered/contract.json')
torch.set_num_threads(3)
net=AutoModel.from_pretrained(reg['retriever'],local_files_only=True).eval()
tok=AutoTokenizer.from_pretrained(reg['retriever'],local_files_only=True)

@torch.inference_mode()
def embed(texts):
    batch=tok(texts,padding=True,truncation=True,return_tensors='pt')
    h=net(**batch).last_hidden_state
    mask=batch['attention_mask'].unsqueeze(-1)
    return (h*mask).sum(1)/mask.sum(1)

banks=read(P.parent/'remastered/banks.json')
for tag in ['4b','9b']:
    rows=list(map(json.loads,(P/(tag+'_rows.jsonl')).read_text().splitlines()))
    assert len(rows)==96
    paths={}
    for bi,b in enumerate(banks):
        clozes={}
        for doc in b['documents']:
            content=doc['text'].removesuffix(' '+doc['obj'])
            assert doc['subject'] in content
            clozes[content.replace(doc['subject'],'{}',1)]=doc['rel']
        names=sorted(clozes)
        mat=embed(names)
        for row in [x for x in rows if x['bank']==bi]:
            labels=[s.strip() for s in re.findall(r'\(([^()]+)\)',row['raw']) if s.strip().casefold()!='answer']
            if labels:
                scores=embed(labels)@mat.T
                vals,inds=scores.max(-1)
                paths[row['id']]=[dict(label=s,rel=clozes[names[i]] if v>=.845 else None,score=float(v)) for s,v,i in zip(labels,vals,inds,strict=True)]
            else:
                paths[row['id']]=[]
    (P/(tag+'_semantic_paths.json')).write_text(json.dumps(paths,ensure_ascii=False)+'\n')
