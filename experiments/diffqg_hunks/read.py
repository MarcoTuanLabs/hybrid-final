"""Evaluate answer-span support under local, reversible edits to each retrieved passage."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from edits import changes,worlds

HERE=Path(__file__).resolve().parent

def normalized(s):
    return ' '.join(re.findall(r'[^\W_]+',s.casefold()))

class SpanScores:
    def __init__(self,text,encoding,n,res):
        offsets=encoding.offset_mapping[n].tolist()
        start=res.start_logits[n].numpy();end=res.end_logits[n].numpy()
        null=float(start[0]+end[0])
        valid=[i for i,v in enumerate(encoding.sequence_ids(n)) if v==1 and offsets[i][1]>offsets[i][0]]
        spans=[]
        for a,i in enumerate(valid):
            for j in valid[a:a+24]:
                l,r=offsets[i][0],offsets[j][1];surface=text[l:r].strip()
                spans.append((float(start[i]+end[j])-null,l,r,surface))
        self.spans=sorted(spans,key=lambda s:-s[0]);self.lookup={}
        for score,l,r,span in self.spans:
            key=normalized(span)
            if key and key not in self.lookup:self.lookup[key]=score

    def score(self,surface):
        return max(-10.,self.lookup.get(normalized(surface),-10.))

    def overlapping(self,h):
        out=[];seen=set()
        for span in self.spans:
            _,l,r,s=span
            if h.new_start==h.new_end or l>=h.new_end or r<=h.new_start:continue
            key=normalized(s)
            if key and key not in seen:out.append(span);seen.add(key)
            if len(out)==4:break
        return out

class Reader:
    def __init__(self,path):
        with (path/'model.safetensors').open('rb') as file:
            assert hashlib.file_digest(file,'sha256').hexdigest()=='ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7'
        torch.set_num_threads(4)
        self.net=AutoModelForQuestionAnswering.from_pretrained(path,local_files_only=True).eval()
        self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.positions=0;self.passages=0

    @torch.inference_mode()
    def __call__(self,question,texts):
        enc=self.tok([question]*len(texts),texts,return_offsets_mapping=True,padding=True,return_tensors='pt',
                     max_length=512,truncation='only_second')
        self.positions+=int(enc.attention_mask.sum());self.passages+=len(texts)
        res=self.net(input_ids=enc.input_ids,attention_mask=enc.attention_mask)
        return [SpanScores(text,enc,i,res) for i,text in enumerate(texts)]

def clip(value):
    return min(6.,max(-6.,value))

def extract(reader,question,row):
    old,new=row['old'],row['current']
    before,after=reader(question,[old,new])
    hunks=[]
    for h in changes(old,new):
        options=after.overlapping(h)
        if options:hunks.append((options[0][0],h,options))
    hunks.sort(key=lambda entry:-entry[0]);hunks=hunks[:2]
    initial=after.spans[0]
    slots={m:(initial[0],initial[3]) for m in ('current','whole','rollback','symmetric')}
    slots['changed']=(-1.e10,'')
    def keep(mode,n,s):
        if n>slots[mode][0]:slots[mode]=(n,s)
    keep('whole',initial[0]+.5*clip(after.score(initial[3])-before.score(initial[3])),initial[3])
    candidates={initial[3]};details=[]
    if hunks:
        variants=reader(question,[txt for _,h,_ in hunks for txt in worlds(old,new,h)])
        for j,(_,h,options) in enumerate(hunks):
            reverted,added=variants[2*j:2*j+2]
            local=[]
            for raw,_,_,s in options:
                candidates.add(s)
                t=after.score(s);b=before.score(s);v=reverted.score(s);a=added.score(s)
                keep('changed',raw,s);keep('whole',raw+.5*clip(t-b),s)
                keep('rollback',raw+.5*clip(t-v),s)
                correction=clip(min(t-v,a-b))+min(0.,clip(a-t))
                keep('symmetric',raw+.5*correction,s)
                local.append(dict(s=s,target=t,base=b,reverted=v,added=a))
            details.append(dict(hunk=[h.old_start,h.old_end,h.new_start,h.new_end],candidates=local))
    if not slots['changed'][1]:slots['changed']=slots['current']
    return dict(policy=slots,proposals=sorted(candidates),hunks=details)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',type=Path);a=parser.parse_args()
    reader=Reader(a.model)
    read=lambda filename:json.loads((HERE/filename).read_text())
    bank={row['key']:row for row in read('bank.json')};ranks=read('ranks.json');qs=read('questions.json')
    start=time.monotonic()
    with (HERE/'span_rows.jsonl').open('w') as f:
        for item in qs:
            uid=item['id'];keys=ranks[uid]['main'];tick=time.monotonic();n=reader.positions;c=reader.passages
            question=f"As of {item['as_of']}, {item['question']}"
            docs={key:extract(reader,question,bank[key]) for key in keys}
            answers={};winners={}
            for mode in docs[keys[0]]['policy']:
                i,key=max(enumerate(keys),key=lambda z:docs[z[1]]['policy'][mode][0]-math.log1p(z[0]))
                winners[mode]=key;answers[mode]=docs[key]['policy'][mode][1]
            f.write(json.dumps(dict(id=uid,answers=answers,winners=winners,docs=docs,seconds=time.monotonic()-tick,
                                    encoder_positions=reader.positions-n,passages=reader.passages-c),ensure_ascii=False)+'\n');f.flush()
            if time.monotonic()-start>1200:raise TimeoutError('twenty-minute CPU screen')
if __name__=='__main__':main()
