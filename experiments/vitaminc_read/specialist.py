"""Run the VitaminC authors' ALBERT classifiers on exactly the paired pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AlbertTokenizer,AlbertForSequenceClassification
from data import split,manifest,LABELS

HERE=Path(__file__).resolve().parent
MODELS={"albert-base-vitaminc-mnli":"5c9fea6e2ffa1fe7187fc3d29ffeba5d77b46da9","albert-xlarge-vitaminc":"e86807a8bfa4f9cfec822c5855f1c86c4f475f03"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("data",type=Path);ap.add_argument("model",type=Path);ap.add_argument("part",choices=("fit","pilot","confirmation"));ap.add_argument("output",type=Path)
    args=ap.parse_args();m=args.model.name;rev=MODELS[m]; torch.set_num_threads(4)
    model=AlbertForSequenceClassification.from_pretrained(args.model,local_files_only=True).eval()
    assert [model.config.id2label[k] for k in range(3)]==LABELS
    tokenizer=AlbertTokenizer.from_pretrained(args.model,local_files_only=True)
    model=model.to('mps');parts=split(args.data)
    assert manifest(parts)==json.loads((HERE/"split.json").read_text())
    rows=[];begin=time.monotonic()
    with torch.inference_mode():
        for r in parts[args.part]:
            # Publisher evaluation writes text_a under evidence and text_b under claim.
            passages=r["passages"]*2;claims=[c for c in r["claims"] for _ in range(2)]
            batch=tokenizer(passages,claims,padding=True,truncation=True,max_length=256,return_tensors="pt").to('mps')
            logits=model(**batch).logits.cpu().reshape(2,2,3)
            rows.append(dict(id=r['id'],page=r['page'],gold=r['gold'],logits=logits.tolist(),answers=logits.argmax(-1).tolist()))
            if time.monotonic()-begin>600:raise TimeoutError(m)
    args.output.mkdir(exist_ok=True,parents=True)
    (args.output/(m+'-'+args.part+'.jsonl')).write_text(''.join(json.dumps(z)+'\n' for z in rows))
    h=hashlib.sha256()
    with (args.model/'model.safetensors').open('rb') as f:h=hashlib.file_digest(f,'sha256')
    print(json.dumps(dict(model=m,revision=rev,weight_sha256=h.hexdigest(),correct=sum(int(p==g) for r in rows for a,b in zip(r['answers'],r['gold']) for p,g in zip(a,b)),total=4*len(rows),seconds=time.monotonic()-begin)),flush=True)
if __name__=="__main__":main()
