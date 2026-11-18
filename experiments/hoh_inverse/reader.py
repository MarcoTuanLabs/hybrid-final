"""Strong current-only extractive comparator on exactly the six new candidate passages."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from transformers import AutoModelForQuestionAnswering,AutoTokenizer

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'hoh_updates'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    with (a.model/'model.safetensors').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()=='ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7'
    torch.set_num_threads(3)
    model=AutoModelForQuestionAnswering.from_pretrained(a.model,local_files_only=True).eval()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    pool=json.loads((HERE/'pool.json').read_text());qs=json.loads((SOURCE/'questions.json').read_text())
    results={};start=time.monotonic()
    with torch.inference_mode():
        for q in qs:
            rows=pool[q['id']];source=[r['current'] for r in rows];tick=time.monotonic()
            inp=tok([q['question']]*len(rows),source,return_offsets_mapping=True,padding=True,return_tensors='pt',
                    truncation='only_second',max_length=384)
            res=model(input_ids=inp.input_ids,attention_mask=inp.attention_mask)
            details={}
            for n,r in enumerate(rows):
                st=res.start_logits[n].numpy();end=res.end_logits[n].numpy()
                off=inp.offset_mapping[n].tolist();valid=[i for i,s in enumerate(inp.sequence_ids(n)) if s==1 and off[i][1]>off[i][0]]
                score,i,j=max((float(st[i]+end[j]),-i,-j) for i in valid for j in valid if i<=j<i+24)
                details[r['key']]=dict(answer=source[n][off[-i][0]:off[-j][1]].strip(),margin=score-float(st[0]+end[0]))
            winner=max(rows,key=lambda r:details[r['key']]['margin']-math.log1p(rows.index(r)))
            results[q['id']]=dict(answers=details,selected=winner['key'],seconds=time.monotonic()-tick)
    (HERE/'spans.json').write_text(json.dumps(results,ensure_ascii=False)+'\n')
    print(len(results),time.monotonic()-start)
if __name__=='__main__':main()
