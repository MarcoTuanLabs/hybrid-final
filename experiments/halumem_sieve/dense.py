"""Measure whole-turn centrality with an independently pretrained dense encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoModel,AutoTokenizer

HERE=Path(__file__).resolve().parent
PIN="08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8"


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("model",type=Path);a=ap.parse_args()
    with (a.model/"pytorch_model.bin").open("rb") as fp:assert hashlib.file_digest(fp,"sha256").hexdigest()==PIN
    torch.set_num_threads(3)
    net=AutoModel.from_pretrained(a.model,local_files_only=True).eval();tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    cases=json.loads((HERE/"raw_inputs.json").read_text());out={}
    for k,row in cases.items():
        tick=time.perf_counter();vectors=[];positions=0
        for i in range(0,len(row["turns"]),12):
            batch=tok([t["text"] for t in row["turns"][i:i+12]],truncation=True,max_length=256,padding=True,return_tensors="pt")
            mask=batch.attention_mask.unsqueeze(-1);positions+=batch.input_ids.numel()
            vectors.append((net(**batch).last_hidden_state*mask).sum(1)/mask.sum(1))
        x=torch.nn.functional.normalize(torch.cat(vectors),dim=-1)
        center=torch.nn.functional.normalize(x.mean(0),dim=0)
        out[k]=dict(scores=(x@center).tolist(),seconds=time.perf_counter()-tick,padded_tokens=positions)
    (HERE/"dense_scores.json").write_text(json.dumps(out)+"\n")


if __name__=="__main__":main()
