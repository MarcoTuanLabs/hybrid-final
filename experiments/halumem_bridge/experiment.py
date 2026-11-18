"""Apply a delayed full dialogue before asking two unseen questions in the same cache."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import numpy as np

HERE=Path(__file__).resolve().parent
MARK="<|im_end|>\n<|im_start|>user\n"
RULE="Use the dated original conversation and retrieved excerpts to answer briefly and accurately. Respect the original timestamps, including sessions received late. If the evidence is insufficient, say unknown.\n\nDated dialogue:\n"
PIN="08b88f3a3697877345669405c51a23f53ed90aa2bab441cd7b7b08659925eef8"
FRONT=32
BACK=48


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now():
    mx.synchronize()
    return time.perf_counter()


def logical_bytes(cache):
    total=0
    for c in cache:
        if hasattr(c,"keys_and_values"):
            if c.keys is not None:total+=sum(v.nbytes for v in c.keys_and_values())
        else:total+=sum(v.nbytes for v in c.cache if v is not None)
    return total


def move(eng,c,ids):
    for i in range(0,len(ids),192):
        eng.body(ids[i:i+192],c)
    mx.eval(*[l.state for l in c])


def date_text(s):
    return "\n".join(f"{t['timestamp']} | {t['role']} | {t['content']}" for t in s.get("dialogue",[]))+"\n"


def unrotate(rope,k,stop):
    # Traversing reversed positions applies -stop+1,...,-start using the same RoPE kernel.
    return rope(k[:,:,::-1,:],offset=1-stop)[:,:,::-1,:]


def correction(old,new,length):
    diff=new.astype(mx.float32)-old.astype(mx.float32)
    early=mx.mean(diff[:,:,:FRONT//2,:],axis=2,keepdims=True)
    late=mx.mean(diff[:,:,FRONT//2:,:],axis=2,keepdims=True)
    # Least-squares decay between the two anchor means, bounded to a stable continuation.
    fac=mx.clip(mx.sum(early*late,axis=-1,keepdims=True)/(1e-8+mx.sum(early*early,axis=-1,keepdims=True)),0.,.99)
    offset=(mx.arange(length,dtype=mx.float32)+8.5)/(FRONT/2)
    return late*mx.power(fac,offset[None,None,:,None])


def corrected_splice(eng,bound,new,oldmid,splice,do_correct):
    result=splice(eng.model,oldmid,bound,new,[],"stale")
    work=0
    if do_correct:
        for i,layer in enumerate(eng.model.model.layers):
            if layer.is_linear:continue
            origin,b=bound[i].offset,new[i].offset
            n=oldmid[i].offset-origin
            rope=layer.self_attn.rope
            ko,vo=bound[i].keys_and_values();kn,vn=new[i].keys_and_values()
            ok,nk=unrotate(rope,ko[:,:,origin-FRONT:origin,:],origin),unrotate(rope,kn[:,:,b-FRONT:b,:],b)
            dk=correction(ok,nk,n)
            dv=correction(vo[:,:,origin-FRONT:origin,:],vn[:,:,b-FRONT:b,:],n)
            keys,values=result[i].keys_and_values()
            keys=mx.concatenate([keys[:,:,:b,:],(keys[:,:,b:,:].astype(mx.float32)+rope(dk,offset=b)).astype(keys.dtype)],axis=2)
            values=mx.concatenate([values[:,:,:b,:],(values[:,:,b:,:].astype(mx.float32)+dv).astype(values.dtype)],axis=2)
            result[i].state=(keys,values,b+n)
            # Separate conservative scalar allowance for projections, rotation, and additions.
            work+=(n+FRONT)*int(keys.shape[1])*int(keys.shape[3])*24
        mx.eval(*[c.state for c in result])
    return result,work


def evidence(index,worker,event,q,tok):
    global_hits=index.search(q,24)
    event_hits=index.search(q,8,only_session=event)
    entries={x[1]:x[2] for x in global_hits+event_hits}
    keys=list(entries)
    worker.stdin.write(json.dumps(dict(question=q,texts=list(entries.values())))+"\n")
    worker.stdin.flush()
    reply=json.loads(worker.stdout.readline())
    dense={keys[idx]:rank+1 for rank,idx in enumerate(reply["order"])}
    lex={x[1]:rank+1 for rank,x in enumerate(global_hits)}
    ranked=sorted(keys,key=lambda key:-(1/(60+dense[key])+(1/(60+lex[key]) if key in lex else 0)))
    selected=sorted(ranked[:4])
    lines=[tok.decode(tok.encode(entries[k],add_special_tokens=False)[:128]) for k in selected]
    return lines,selected,reply["positions"],reply["macs"]


def question(eng,c,q,lines):
    actual="Dated evidence:\n"+"\n".join(lines)+"\n\nQuestion: "+q
    template=eng.tok.apply_chat_template([dict(role="system",content=RULE),dict(role="user",content=actual)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=eng.tok.encode(MARK+template.split(MARK,1)[1],add_special_tokens=False)
    move(eng,c,ids[:-1])
    logits=eng.body(ids[-1:],c,answer=True)
    tokens=[]
    for _ in range(48):
        t=int(mx.argmax(logits).item())
        if t in eng.tok.eos_token_ids:break
        tokens.append(t)
        if len(tokens)==48 or "\n" in eng.tok.decode(tokens):
            move(eng,c,[t])
            break
        logits=eng.body([t],c,answer=True)
    return tokens


def run(eng,worker,Index,splice,sessions,event,at,questions,mode):
    eng.reset();start=now();idx=Index();idx.add(sessions,[i for i in range(at+1) if i!=event])
    enc=lambda s:eng.tok.encode(s,add_special_tokens=False)
    system=eng.tok.apply_chat_template([dict(role="system",content=RULE),dict(role="user",content="Later")],tokenize=False,add_generation_prompt=True,enable_thinking=False).split(MARK,1)[0]
    shell=enc(system)
    prefix=make_prompt_cache(eng.model)
    repair=mode in ("splice","bridge")
    if mode!="fresh":
        prior=enc(date_text(sessions[event-1]))[-128:]
        all_tail=enc("".join(date_text(sessions[j]) for j in range(event+1,at+1)))
        assert len(all_tail)>2048
        tail=all_tail[:1024]+all_tail[-1024:]
        move(eng,prefix,shell+prior)
        if repair:
            bound=copy.deepcopy(prefix);move(eng,bound,tail[:FRONT])
            oldmid=copy.deepcopy(bound);move(eng,oldmid,tail[FRONT:-BACK])
            old=copy.deepcopy(oldmid);move(eng,old,tail[-BACK:])
            storage=sum(logical_bytes(c) for c in (prefix,bound,oldmid,old))
        else:
            old=copy.deepcopy(prefix);move(eng,old,tail)
            storage=logical_bytes(old)+(logical_bytes(prefix) if mode=="checkpoint" else 0)
            if mode!="checkpoint":del prefix
    else:
        move(eng,prefix,shell);storage=logical_bytes(prefix)
    ingest=now();ingest_pos=eng.positions;ingest_macs=eng.macs
    idx.add(sessions,[event]);scalar_ops=0;late_n=0
    if mode=="fresh":live=prefix
    elif mode=="old_rag":live=old
    elif mode=="notice":
        live=old;late=enc("\nDated session received late. Its original date applies.\n"+date_text(sessions[event]));late_n=len(late)
        move(eng,live,late)
    else:
        late=enc(date_text(sessions[event]));late_n=len(late)
        new=prefix
        if mode=="checkpoint":
            move(eng,new,late+tail);live=new
        else:
            move(eng,new,late+tail[:FRONT])
            live,scalar_ops=corrected_splice(eng,bound,new,oldmid,splice,mode=="bridge")
            move(eng,live,tail[-BACK:])
    ready=now();update_pos=eng.positions-ingest_pos
    search_s=0;bert_pos=0;bert_macs=0;selected=[];tokens=[]
    for q in questions:
        st=now();lines,hits,n,cost=evidence(idx,worker,event,q,eng.tok)
        search_s+=now()-st;bert_pos+=n;bert_macs+=cost;selected.append(hits)
        tokens.append(question(eng,live,q,lines))
    end=now()
    return dict(replies=[eng.tok.decode(ts).strip() for ts in tokens],tokens=tokens,selected=selected,
                setup_s=ingest-start,update_s=ready-ingest,search_s=search_s,cold_s=end-start,online_s=end-ingest,
                decoder_macs=eng.macs,encoder_macs=bert_macs,encoder_positions=bert_pos,decoder_positions=eng.positions,
                setup_positions=ingest_pos,setup_macs=ingest_macs,update_positions=update_pos,bridge_scalar_ops=scalar_ops,
                logical_old_cache_bytes=storage,delayed_model_tokens=late_n,suffix_tokens=0 if mode=="fresh" else len(tail),
                skipped_tokens=len(tail)-FRONT-BACK if repair else 0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("data","model","public","retriever"):ap.add_argument(arg,type=Path)
    ap.add_argument("--reranker-python",default=sys.executable)
    a=ap.parse_args();repo=a.public.resolve();exper=repo/"experiments"
    parent=exper/"halumem_feasibility"
    rt=imported("halu_raw_index",parent/"run_text.py")
    engine=imported("chronological_engine",exper/"late_arrival/retention_splice/run.py")
    manifest=json.loads((exper/"read_projection/release.json").read_text())
    for name,h in manifest["model_files"].items():
        with (a.model/name).open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==h
    with a.data.open("rb") as f:raw=[f.readline() for _ in range(4)]
    audit=json.loads((parent/"audit.json").read_text())
    assert hashlib.sha256(b"".join(raw)).hexdigest()==audit["input_first_four_digest"]
    users=[json.loads(row) for row in raw]
    cases=json.loads((parent/"observations.json").read_text());assert len(cases)==8
    cfg=json.loads((HERE/"contract.json").read_text())
    env=os.environ.copy()
    if "RERANK_PYTHONPATH" in env:env["PYTHONPATH"]=env["RERANK_PYTHONPATH"]
    with subprocess.Popen([a.reranker_python,str(HERE/"rerank.py"),str(a.retriever),PIN],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env) as worker:
        try:
            assert json.loads(worker.stdout.readline())["ready"]
            model,tok=load(str(a.model));mx.set_cache_limit(96*1024**2);eng=engine.Engine(model,tok)
            eng.reset();move(eng,make_prompt_cache(model),[1,2])
            dst=HERE/"rows.jsonl"
            seen={tuple(json.loads(row)["key"]) for row in dst.read_text().splitlines()} if dst.exists() else set()
            elapsed=now()
            with dst.open("a") as out:
                for record in cases:
                    uid,event,at=record["key"];s=users[uid]["sessions"]
                    qs=[s[at]["questions"][j]["question"] for j in record["question_indices"]]
                    modes=list(cfg["methods"]);random.Random(1857+uid*43+event).shuffle(modes)
                    for mode in modes:
                        key=(uid,event,at,mode)
                        if key in seen:continue
                        row=run(eng,worker,rt.Index,engine.splice,s,event,at,qs,mode)
                        out.write(json.dumps(dict(key=key,question_indices=record["question_indices"],**row))+"\n");out.flush()
                        print(key,round(row["cold_s"],2),round(now()-elapsed,1),flush=True)
                        mx.clear_cache()
                        if now()-elapsed>3300:raise TimeoutError("55-minute screen")
        finally:
            worker.stdin.close();worker.wait(timeout=30)


if __name__=="__main__":
    main()
