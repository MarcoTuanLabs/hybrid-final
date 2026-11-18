"""Select evidence per edition using lexical, dense and revision-difference information."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from transformers import AutoModel,AutoTokenizer

P=Path(__file__).resolve().parent


def words(s):
    return re.findall(r'[a-z0-9]+',s.casefold())


def candidates(edition):
    seen=set()
    result=[]
    for r in edition['rows']:
        text=' | '.join(x for x in [r['section'],*r['cells']] if x)
        text=' '.join(text.split()[:65])
        if text and text not in seen:
            result.append(dict(text=text,cells=r['cells'],section=r['section']))
            seen.add(text)
    for para in edition['paragraphs']:
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z\[])',para):
            text=' '.join(sentence.split()[:60])
            if text and text not in seen:
                result.append(dict(text=text,cells=[],section=''))
                seen.add(text)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights',type=Path);args=parser.parse_args()
    torch.set_num_threads(4)
    with (args.weights/'pytorch_model.bin').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()=='08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8'
    model=AutoModel.from_pretrained(args.weights,local_files_only=True).eval()
    tok=AutoTokenizer.from_pretrained(args.weights,local_files_only=True)
    @torch.inference_mode()
    def embed(texts):
        chunks=[]
        for start in range(0,len(texts),24):
            ins=tok(texts[start:start+24],max_length=128,truncation=True,padding=True,return_tensors='pt')
            mask=ins['attention_mask'].unsqueeze(-1)
            chunks.append((model(**ins).last_hidden_state*mask).sum(1)/mask.sum(1))
        return torch.cat(chunks)
    out=[]
    start=time.monotonic()
    for item in json.loads((P/'inputs.json').read_text()):
        q=item['question'];page=item['page']
        terms=set(words(q))-set(words(page))-set('what is who are of how the in does did do for play many which a an s currently as by and at was were or'.split())
        docs=[candidates(e) for e in item['editions']]
        rowsets=[{r['text'].casefold() for r in side} for side in docs]
        dense_q=embed([q])[0]
        group=[]
        for side,rows in enumerate(docs):
            assert rows,item['page']
            counts=[Counter(words(r['text'])) for r in rows]
            length=sum(sum(c.values()) for c in counts)/len(counts)
            df={t:sum(t in c for c in counts) for t in terms}
            bm=[sum(math.log(1+(len(rows)-df[t]+.5)/(df[t]+.5))*2.2*c[t]/(c[t]+1.2*(.25+.75*sum(c.values())/length)) for t in terms if c[t]) for c in counts]
            dense=(embed([r['text'] for r in rows])@dense_q).tolist()
            def ranks(scores):
                order=sorted(range(len(rows)),key=lambda i:(-scores[i],i))
                return {ix:j for j,ix in enumerate(order)}
            lex,sem=ranks(bm),ranks(dense)
            def score(i):
                changed=rows[i]['text'].casefold() not in rowsets[1-side]
                return (1/(5+lex[i])+1/(5+sem[i]))*(1.20 if changed else 1.)
            ranked=sorted(range(len(rows)),key=lambda i:(-score(i),i))
            group.append([dict(**rows[i],source_index=i) for i in ranked[:5]])
        out.append(dict(id=item['id'],page=page,question=q,dates=item['dates'],evidence=group,
                        revisions=[e['revision'] for e in item['editions']]))
    (P/'ranked.json').write_text(json.dumps(out,ensure_ascii=False)+'\n')
    (P/'rank_receipt.json').write_text(json.dumps(dict(n=len(out),wall_s=time.monotonic()-start))+'\n')
if __name__=='__main__':main()
