"""Run the original fixed memory reader on two public CAPITALS templates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np
from inputs import example,load_inputs

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("phase",choices=("dev","test"))
    for arg in ("data","model","public"):p.add_argument(arg,type=Path)
    a=p.parse_args();repo=a.public.resolve()
    if a.phase=="test":assert json.loads((HERE/"dev_summary.json").read_text())["promote"]
    sys.path[:0]=[str(repo/"experiments/quantized_write"),str(repo/"experiments/jacobian_service")]
    from run import Trial
    from runtime import stream
    pins=json.loads((repo/"experiments/read_projection/release.json").read_text())
    for k,h in pins["model_files"].items():
        with (a.model/k).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==h
    trial=Trial(a.model,json.loads((repo/"evidence/quantized_9b/contract.json").read_text()))
    mx.set_cache_limit(96*1024**2)
    data=load_inputs(a.data,trial.tok)
    out=HERE/(a.phase+"_rows.jsonl")
    seen={tuple(json.loads(l)["key"]) for l in out.read_text().splitlines()} if out.exists() else set()
    start=time.monotonic()
    seeds=range(51973000,51973012) if a.phase=="dev" else range(51974000,51974064)
    with out.open("a") as f:
        for seed in seeds:
            for fmt in ("DEFAULT","WORK"):
                for nuisance in (False,True):
                    key=(seed,fmt,nuisance)
                    if key in seen:continue
                    case=example(data,trial.tok,*key)
                    y0,y1=case["recipient"],case["alternate"]
                    tail=case["suffix"]
                    def metrics(logits):
                        return dict(top=int(np.argmax(logits)),margin=float(logits[y1]-logits[y0]))
                    left=trial.prefill(case["a"]);right=trial.prefill(case["b"])
                    edit=trial.write(left,right)
                    ref={}
                    for name,base in (("recipient",left),("donor",right),("native",edit)):
                        logits,_=stream(trial,base,base,tail,True,"batch_native")
                        ref[name]=metrics(logits)
                    broad=copy.deepcopy(left)
                    c=broad[28][0]
                    lo=2*trial.model.model.layers[28].linear_attn.key_dim
                    row=mx.concatenate([c[:,-1:,:lo],right[28][0][:,-1:,lo:]],axis=-1)
                    broad[28][0]=mx.concatenate([c[:,:-1],row],axis=1)
                    ref["wide_native"]=metrics(stream(trial,left,broad,tail,True,"batch_native")[0])
                    for m,solver in (("white","batch_white"),("normal","batch_observed")):
                        for active in (False,True):
                            logits,_=stream(trial,left,edit,tail,active,solver)
                            ref[m if active else m+"_unwritten"]=metrics(logits)
                    record=dict(key=key,recipient=y0,alternate=y1,outputs=ref,source_pair=case["pair_sha256"],
                                prefix_tokens=len(case["a"]),wide_coordinates=int(row.shape[-1])-lo)
                    f.write(json.dumps(record)+"\n");f.flush()
            mx.clear_cache();print(seed,round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>(480 if a.phase=="dev" else 2400):raise TimeoutError(a.phase)


if __name__=="__main__":main()
