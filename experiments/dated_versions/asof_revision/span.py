"""Run a conventional extractive question-answerer over only the requested edition."""
import json
from pathlib import Path
import sys
import time
import torch
from transformers import AutoTokenizer,AutoModelForQuestionAnswering

P=Path(__file__).resolve().parent
torch.set_num_threads(4)
tokenizer=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True)
model=AutoModelForQuestionAnswering.from_pretrained(sys.argv[1],local_files_only=True).eval()
start=time.monotonic();out={}
for item in json.loads((P/'ranked.json').read_text()):
    pair=[]
    for evidence in item['evidence']:
        context='\n'.join(row['text'] for row in evidence)
        x=tokenizer(item['question'],context,return_offsets_mapping=True,return_tensors='pt',max_length=512,truncation='only_second')
        ids=x.sequence_ids(0);offset=x.pop('offset_mapping')[0].tolist()
        with torch.inference_mode(): scores=model(**x)
        a,b=scores.start_logits[0],scores.end_logits[0]
        opts=[(float(a[i]+b[j]),i,j) for i in range(len(ids)) if ids[i]==1 for j in range(i,min(i+32,len(ids))) if ids[j]==1]
        _,i,j=max(opts);pair.append(context[offset[i][0]:offset[j][1]])
    out[item['id']]=pair
(P/'span_answers.json').write_text(json.dumps(out,ensure_ascii=False)+'\n')
(P/'span_receipt.json').write_text(json.dumps(dict(wall_s=time.monotonic()-start,n=len(out)))+'\n')
