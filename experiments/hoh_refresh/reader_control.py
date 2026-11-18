"""Measure a direct two-edition CPU span reader on the same page subscriptions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',type=Path);a=parser.parse_args()
    with (a.model/'model.safetensors').open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()=='ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7'
    torch.set_num_threads(4)
    model=AutoModelForQuestionAnswering.from_pretrained(a.model,local_files_only=True).eval()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    items=json.loads((HERE/'inputs.json').read_text())
    with (HERE/'reader_rows.jsonl').open('w') as f,torch.inference_mode():
        for row in items:
            tick=time.monotonic();answers={};positions=0
            for field in ['old','current']:
                passage=row[field]
                inp=tok(row['question']+' (Page: '+row['title']+')',passage,return_offsets_mapping=True,
                        return_tensors='pt',truncation='only_second',max_length=384)
                raw=model(input_ids=inp.input_ids,attention_mask=inp.attention_mask)
                positions+=int(inp.attention_mask.sum())
                start=raw.start_logits[0].numpy();end=raw.end_logits[0].numpy()
                offsets=inp.offset_mapping[0].tolist()
                use=[i for i,s in enumerate(inp.sequence_ids()) if s==1 and offsets[i][1]>offsets[i][0]]
                _,i,j=max((float(start[i]+end[j]),-i,-j) for i in use for j in use if i<=j<i+24)
                answers[field]=passage[offsets[-i][0]:offsets[-j][1]].strip()
            f.write(json.dumps(dict(id=row['id'],answers=answers,seconds=time.monotonic()-tick,
                                    encoder_positions=positions),ensure_ascii=False)+'\n')
if __name__=='__main__':
    main()
