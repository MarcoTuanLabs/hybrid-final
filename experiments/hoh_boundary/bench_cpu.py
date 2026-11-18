"""Time the two-edition span reader and both online surface repair policies."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForQuestionAnswering,AutoTokenizer
from guard import guarded,revise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path);p.add_argument('inputs',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args()
    with (a.model/'model.safetensors').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()=='ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7'
    torch.set_num_threads(4)
    model=AutoModelForQuestionAnswering.from_pretrained(a.model,local_files_only=True).eval()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    items=json.loads(a.inputs.read_text())
    def read(row,key):
        passage=row[key]
        inp=tok(row['question']+' (Page: '+row['title']+')',passage,return_offsets_mapping=True,
                return_tensors='pt',truncation='only_second',max_length=384)
        res=model(input_ids=inp.input_ids,attention_mask=inp.attention_mask)
        st=res.start_logits[0].numpy();en=res.end_logits[0].numpy();off=inp.offset_mapping[0].tolist()
        use=[i for i,s in enumerate(inp.sequence_ids()) if s==1 and off[i][1]>off[i][0]]
        _,i,j=max((float(st[i]+en[j]),-i,-j) for i in use for j in use if i<=j<i+24)
        return passage[off[-i][0]:off[-j][1]].strip(),int(inp.attention_mask.sum())
    modes=['full','unique','guarded_unique'];start=time.monotonic()
    with torch.inference_mode(),a.out.open('w') as out:
        read(items[0],'old')  # Warm the CPU kernels before any measured policy.
        for j,row in enumerate(items):
            policies={}
            for mode in modes[j%3:]+modes[:j%3]:
                tick=time.monotonic();old,n=read(row,'old');old_s=time.monotonic()-tick
                decision='replay';fresh=None;total=n
                if mode!='full':
                    rule=guarded if mode.startswith('guarded_') else revise
                    fresh,decision=rule(row['old'],row['current'],old,'unique',[])
                if fresh is None:
                    fresh,extra=read(row,'current');total+=extra
                policies[mode]=dict(old=old,current=fresh,old_positions=n,total_positions=total,
                                    seconds=time.monotonic()-tick,old_s=old_s,decision=decision)
            out.write(json.dumps(dict(id=row['id'],policies=policies),ensure_ascii=False)+'\n')
            if time.monotonic()-start>1800:raise TimeoutError('registered thirty-minute budget')
if __name__=='__main__':main()
