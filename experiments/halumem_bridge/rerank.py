"""Rank an incoming query and its candidate excerpts in a persistent CPU worker."""
import json
import sys
from pathlib import Path
import hashlib
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    path=Path(sys.argv[1]); pin=sys.argv[2]
    with (path/"pytorch_model.bin").open("rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest()==pin
    torch.set_num_threads(3)
    net=AutoModel.from_pretrained(path,local_files_only=True).eval()
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    cfg=net.config
    @torch.inference_mode()
    def rank(q, texts):
        reps=[];macs=positions=0
        for i in range(0,len(texts)+1,12):
            x=tok(([q]+texts)[i:i+12],padding=True,truncation=True,max_length=160,return_tensors="pt")
            mask=x.attention_mask.unsqueeze(-1)
            reps.append((net(**x).last_hidden_state*mask).sum(1)/mask.sum(1))
            b,n=x.input_ids.shape;h=cfg.hidden_size
            positions+=b*n
            macs+=b*cfg.num_hidden_layers*(n*(4*h*h+2*h*cfg.intermediate_size)+2*n*n*h)
        v=torch.cat(reps)
        return dict(order=torch.argsort(v[1:]@v[0],descending=True,stable=True).tolist(),positions=positions,macs=macs)
    rank("query",["text"])
    print(json.dumps(dict(ready=True)),flush=True)
    for line in sys.stdin:
        job=json.loads(line)
        print(json.dumps(rank(job["question"],job["texts"])),flush=True)


if __name__=="__main__":
    main()
