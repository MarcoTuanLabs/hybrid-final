"""Apply one pre-question update and carry the same conversational cache through three turns."""
import argparse
import copy
from collections import Counter
import difflib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'shared_read_basis'))
from documents import MARK,document,encoding,message,setup,template
from cache_service import advance,copy_at,elapsed,size
from wire import apply_patch,decode_state,encode_state,patch,prefix
from stream import transfer
# Load the shared-document implementation explicitly: both experiments expose serve.py.
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('shared_document_service',HERE.parent/'shared_read_basis/serve.py');shared=module_from_spec(spec);spec.loader.exec_module(shared)

def notice(old,new):
    lines=[]
    for op,i,j,a,b in difflib.SequenceMatcher(a=old,b=new,autojunk=False).get_opcodes():
        if op=='equal':continue
        if op in ['delete','replace']:lines.append('Superseded: '+' | '.join(old[i:j]))
        if op in ['insert','replace']:lines.append('Now in effect, in order: '+' | '.join(new[a:b]))
    return '\n\nUpdate notice (latest version).\n'+'\n'.join(lines)
def terms(text):
    return re.findall(r'[a-z0-9]+',text.lower())
def retrieve(facts,q):
    qwords=set(terms(q))-set('where what which is the in would of to did does for a now'.split())
    corpus=[set(terms(s)) for s in facts];df=Counter(w for row in corpus for w in row)
    def score(j):return sum(math.log(1+(len(facts)-df[w]+.5)/(df[w]+.5)) for w in qwords&corpus[j])
    # Recency breaks lexical ties; return selected statements in original episode order.
    ids=sorted(sorted(range(len(facts)),key=lambda j:(-score(j),-j))[:4])
    return [facts[j] for j in ids]
def turn(trial,c,q,limit):
    suffix=encoding(trial,MARK+template(trial,'Shared document.',q).split(MARK,1)[1])
    llm=trial.model.language_model
    def next_token(ids):
        h=trial.model.model(mx.array([ids]),cache=c)[:,-1:,:]
        logits=llm.model.embed_tokens.as_linear(h) if llm.args.tie_word_embeddings else llm.lm_head(h)
        return int(mx.argmax(logits).item())
    token=next_token(suffix);tokens=[]
    for _ in range(limit):
        if token in trial.tok.eos_token_ids:break
        tokens.append(token)
        token=next_token([token])
        if '\n' in trial.tok.decode(tokens):break
    return tokens
def checkpoints(trial,ids,keep):
    c=make_prompt_cache(trial.model);saved={0:copy.deepcopy(c)}
    for i in range(0,len(ids),32):
        part=ids[i:i+32];advance(trial.model,c,part)
        if keep:saved[i+len(part)]=copy.deepcopy(c)
    return c,saved
def bounded(trial,old,cp,old_ids,ids):
    c0=prefix(old_ids,ids);tail=prefix(old_ids[c0:][::-1],ids[c0:][::-1]);skip=max(0,tail-16)
    c,cut=copy_at(cp,c0);advance(trial.model,c,ids[cut:len(ids)-skip])
    if not skip:return c,0
    for j,layer in enumerate(trial.model.model.layers):
        if layer.is_linear:c[j]=copy.deepcopy(old[j]);continue
        n=c[j].offset;old_at=old[j].offset-skip
        keys=old[j].keys[:,:,old_at:old[j].offset,:];values=old[j].values[:,:,old_at:old[j].offset,:]
        shift=n-old_at
        if shift:
            shape=keys.shape;keys=layer.self_attn.rope(keys.reshape(-1,1,1,shape[-1]),offset=shift).reshape(shape)
        c[j].keys=mx.concatenate([c[j].keys[:,:,:n],keys],axis=2);c[j].values=mx.concatenate([c[j].values[:,:,:n],values],axis=2);c[j].offset=n+skip
    mx.eval(*[l.state for l in c]);return c,skip
def matrix_packet(trial,old,new,basis):
    parts=[]
    for j,(i,l) in enumerate((i,l) for i,l in enumerate(trial.model.model.layers) if l.is_linear):
        d=(new[i][1]-old[i][1])[0];parts.append(('a'+str(i),(d@basis[j]).astype(mx.float16)))
    mx.eval(*[v for _,v in parts])
    return encode_state({},parts)[0]
def increment(trial,old,packet,basis):
    _,parts=decode_state(packet);c=copy.deepcopy(old)
    for j,(i,l) in enumerate((i,l) for i,l in enumerate(trial.model.model.layers) if l.is_linear):
        d=parts['a'+str(i)].astype(mx.float32)@basis[j].swapaxes(-1,-2);c[i][1]=old[i][1]+d[None]
    mx.eval(*[l.state for l in c]);return c
def measure(trial,x,alt,mode,basis,other):
    src=x['source'];want_rag=mode in ['old_notice_rag','native_notice_rag','fresh_rag']
    want_notice=mode in ['old_notice','old_notice_rag','native_notice_rag']
    use_native=mode in ['native','native_notice_rag','bank_shared']
    need_cp=use_native or mode in ['checkpoint','bounded16']
    start=elapsed();old_facts=x['old'];old_raw=json.dumps(old_facts,ensure_ascii=False,separators=(',',':')).encode()
    if mode!='fresh_rag':
        old_ids=document(trial,message(src,old_facts));old,saved=checkpoints(trial,old_ids,need_cp)
    else:
        old_ids=[];old=None;saved={}
    initial_s=elapsed()-start
    storage=size(trial,old) if old is not None else 0
    if need_cp:storage=sum(size(trial,c) for c in saved.values())
    start=elapsed();donor_s=0.;wire=0;skip=0;current=None
    if want_rag or want_notice or mode in ['full','checkpoint','bounded16']:
        # Each such receiver recovers the updated facts from old local bytes and the physical patch.
        new_raw=json.dumps(x['new'],ensure_ascii=False,separators=(',',':')).encode()
        pkt,n=transfer(patch(old_raw,new_raw));wire+=n;current=json.loads(apply_patch(old_raw,pkt));assert current==x['new']
    if use_native:
        tick=elapsed();ids=document(trial,message(src,x['new']));common=prefix(old_ids,ids);donor,cut=copy_at(saved,common)
        advance(trial.model,donor,ids[cut:]);donor_s=elapsed()-tick
        if mode=='bank_shared':
            blob,_=shared.encode_document(trial,old,donor,common,'read',{'read':basis});pkt,n=transfer(blob);wire+=n
            c=shared.restore(trial,old,pkt,'read',{'read':other})
        else:
            pkt,n=transfer(matrix_packet(trial,old,donor,basis));wire+=n;c=increment(trial,old,pkt,other)
    elif mode in ['old','old_notice','old_notice_rag']:c=copy.deepcopy(old)
    elif mode=='fresh_rag':
        c=make_prompt_cache(trial.model);advance(trial.model,c,document(trial,'Use the current retrieved facts to answer. Give only the shortest correct answer.'))
    else:
        ids=document(trial,message(src,current))
        if mode=='bounded16':c,skip=bounded(trial,old,saved,old_ids,ids)
        else:
            c,cut=copy_at(saved,prefix(old_ids,ids)) if mode=='checkpoint' else (make_prompt_cache(trial.model),0)
            advance(trial.model,c,ids[cut:])
    if want_notice:advance(trial.model,c,encoding(trial,notice(old_facts,current)))
    ready=elapsed();tokens=[];selected=[]
    for q in [x['q'],x['q_control'],alt]:
        docs=retrieve(current,q) if want_rag else []
        actual=('Current evidence (original order):\n'+'\n'.join(docs)+'\nQuestion: '+q) if docs else q
        tokens.append(turn(trial,c,actual,12 if src=='babi' else 24));selected.append(len(docs))
    end=elapsed();return dict(mode=mode,wire_bytes=wire,old_cache_logical_bytes=storage,old_setup_s=initial_s,ready_cold_s=ready-start,ready_warm_s=ready-start-donor_s,donor_s=donor_s,downstream_s=end-ready,complete_cold_s=initial_s+end-start,complete_warm_s=end-start-donor_s,skipped_suffix=skip,selected=selected,tokens=tokens,texts=[trial.tok.decode(t) for t in tokens])
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('public',type=Path);a=ap.parse_args()
    trial,sets=setup(a.public,a.model);banks=np.load(HERE.parent/'shared_read_basis/bases.npz')
    t=elapsed();basis=mx.array(banks['read']);packet,n=transfer(encode_state({},[('b',basis)])[0]);_,a=decode_state(packet);client=a['b'].astype(mx.float32);basis=basis.astype(mx.float32);mx.eval(basis,client)
    (HERE/'install.json').write_text(json.dumps(dict(seconds=elapsed()-t,wire_bytes=n))+'\n')
    warm=make_prompt_cache(trial.model);advance(trial.model,warm,document(trial,'A short document.'));turn(trial,warm,'What kind of document?',3)
    cases=sets['eval'];rng=random.Random(75366);rng.shuffle(cases)
    paraphrases={(v['source'],v['id']):v['alternate'] for v in json.loads((HERE/'paraphrases.json').read_text())}
    path=HERE/'rows.jsonl';seen={(v['source'],v['id']) for v in map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    start=elapsed()
    with path.open('a') as f:
        for x in cases:
            key=x['source'],x['id'];modes=['old','old_notice','old_notice_rag','fresh_rag','full','checkpoint','bounded16','native','native_notice_rag','bank_shared'];rng.shuffle(modes)
            if key in seen:continue
            arms=[]
            for m in modes:
                arms.append(measure(trial,x,paraphrases[key],m,basis,client));mx.clear_cache()
            f.write(json.dumps(dict(source=key[0],id=key[1],arms=arms))+'\n');f.flush()
            print(key,round(elapsed()-start,1),flush=True)
            if elapsed()-start>1500:raise TimeoutError('Multiturn budget')
if __name__=='__main__':main()
