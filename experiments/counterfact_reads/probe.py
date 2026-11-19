"""Apply equally sized two-head writes after a public factual revision."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen
from mlx_lm.models.gated_delta import compute_g
import numpy as np
from cases import manifest, select
from geometry import designs

HERE = Path(__file__).resolve().parent

def encode(tokenizer, fact, query):
    body = ("Complete the final sentence with the missing word only. Use the supplied fact for its subject, "
            "and your own knowledge for other subjects.\n\nFact: "+fact+".\n\nComplete: "+query)
    text = tokenizer.apply_chat_template([dict(role="user",content=body)], tokenize=False,
                                         add_generation_prompt=True,enable_thinking=False)
    ids = tokenizer.encode(text,add_special_tokens=False)
    boundary = tokenizer.encode("<think>",add_special_tokens=False)
    assert len(boundary) == 1 and ids.count(boundary[0]) == 1
    cut = ids.index(boundary[0])
    assert len(ids)-cut == 4
    return ids[:cut],ids[cut:]

def host(t):
    return np.asarray(t.astype(mx.float32),dtype=np.float64)

def observe(trial, cache, tail):
    state = copy.deepcopy(cache)
    trace = []
    for token in tail:
        ordinal = 0
        def hook(q,k,v,a,b,al,bias,s,mask,**kwargs):
            nonlocal ordinal
            index = ordinal
            ordinal += 1
            y,after = trial.kernel(q,k,v,a,b,al,bias,s,mask,**kwargs)
            if index == trial.ordinal:
                trace.append((q[0,0],k[0,0],mx.sigmoid(b)[0,0],compute_g(al,a,bias)[0,0],after[0]))
            return y,after
        qwen.gated_delta_update = hook
        try:
            logits = trial.model(mx.array([[token]]),cache=state)[0,-1]
            mx.eval(logits)
        finally:
            qwen.gated_delta_update = trial.kernel
    assert len(trace) == 4
    return host(logits),[tuple(host(t) for t in row) for row in trace]

def direct(trial,cache,tail):
    cache = copy.deepcopy(cache)
    for token in tail:
        logits = trial.model(mx.array([[token]]),cache=cache)[0,-1]
        mx.eval(logits)
    return host(logits)

def main():
    p=argparse.ArgumentParser()
    for field in ("model","repo","data","out"):
        p.add_argument(field,type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.repo.resolve()/"experiments/write_window"))
    from service import validate
    trial,_ = validate(a.model,a.repo.resolve())
    sys.path.insert(0,str(a.repo.resolve()/"experiments/read_projection"))
    from codec import observability
    records=select(a.data,trial.tok)
    assert manifest(records)==json.loads((HERE/"cases.json").read_text())
    a.out.mkdir(parents=True,exist_ok=True)
    dest=a.out/"rows.jsonl"
    seen={json.loads(s)["case"] for s in dest.read_text().splitlines()} if dest.exists() else set()
    begin=time.monotonic()
    mx.set_cache_limit(96*1024**2)
    for index,record in enumerate(records):
        if record["case_id"] in seen:
            continue
        fact=record["requested_rewrite"]
        stem=fact["prompt"].format(fact["subject"])
        words=[fact[x]["str"] for x in ("target_true","target_new")]
        gold=[trial.tok.encode(word,add_special_tokens=False)[0] for word in words]
        questions=[stem,record["paraphrase_prompts"][0],record["neighborhood_prompts"][0]]
        items=[]
        for kind,query in zip(("canonical","paraphrase","neighbor"),questions,strict=True):
            prompts=[encode(trial.tok,stem+" "+word,query) for word in words]
            tail=prompts[0][1];assert tail==prompts[1][1]
            old,new=[trial.prefill(z[0]) for z in prompts]
            def outcome(y):
                top=int(y.argmax())
                answer=trial.tok.decode([top]).strip().casefold()
                return dict(top=top,hit=[answer==word.casefold() for word in words],
                            new_minus_old=float(y[gold[1]]-y[gold[0]]))
            base,trace=observe(trial,old,tail)
            outputs={"old":outcome(base),"full":outcome(direct(trial,new,tail))}
            delta=host(new[28][1][0])-host(old[28][1][0])
            count=delta.shape[0]
            queries,w=observability([t[:3] for t in trace],count)
            gates=np.stack([t[3] for t in trace],axis=-1).cumprod(axis=-1)
            w=w*gates[:,None,:]
            states=np.stack([t[4] for t in trace],axis=-1)
            background=np.einsum('hvkt,hkt->hvt',states,queries)
            assert np.isfinite(w).all() and np.isfinite(background).all()
            rec=copy.deepcopy(old)
            for i,layer in enumerate(trial.model.model.layers):
                if layer.is_linear:rec[i][1]=new[i][1]
            outputs["all_recurrent_matrices"]=outcome(direct(trial,rec,tail))
            layer=copy.deepcopy(old);layer[28][1]=new[28][1]
            if index == 0:
                dense,changed_trace=observe(trial,layer,tail)
                changed_state=np.stack([t[4] for t in changed_trace],axis=-1)
                actual=np.einsum('hvkt,hkt->hvt',changed_state-states,queries)
                rel=float(np.linalg.norm(actual-delta@w)/max(np.linalg.norm(actual),1e-12))
                assert rel < 0.005,rel
                outputs["layer_dense"]=dict(outcome(dense),forecast_relative_error=rel)
            else:
                outputs["layer_dense"]=outcome(direct(trial,layer,tail))
            for method,head,update,nbytes in designs(delta,w,background):
                c=copy.deepcopy(old)
                state=host(old[28][1][0]);state[head:head+2]+=update
                c[28][1]=mx.array(state[None],dtype=old[28][1].dtype)
                outputs[method]=dict(outcome(direct(trial,c,tail)),head=head,bytes=nbytes)
            items.append(dict(kind=kind,arms=outputs))
        with dest.open("a") as f:
            f.write(json.dumps(dict(case=record["case_id"],relation=fact["relation_id"],items=items))+"\n")
        mx.clear_cache()
        if (index+1)%4==0:print(index+1,round(time.monotonic()-begin,1),flush=True)
        if time.monotonic()-begin>1200:raise TimeoutError("CounterFact pilot")
if __name__=="__main__":main()
