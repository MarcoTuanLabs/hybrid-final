"""Use the same retrieved edits to compare ordinary and recurrent-channel contrastive reads."""
import copy
import json,hashlib,re,sys,time,random
from pathlib import Path
import os, gzip
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
p=Path(__file__).resolve().parent
def read(name):return json.loads((p/name).read_text())
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def route(q,docs,schema):
    names={norm(d['subject']) for d in docs}
    for rel,temp in schema.items():
        a,b=map(re.escape,temp.strip().split('{}',1));m=re.fullmatch(a+'(.+?)'+b,q.strip(),re.I)
        if not m:continue
        ent=m[1];label=norm(ent)
        if label not in names and (len(label.split())>6 or re.search(r"\b(who|which|where|what|whom|whose|that|of|in|by|for|from|with)\b|['’]s\b",ent,re.I)):continue
        hits={d['obj'] for d in docs if norm(d['subject'])==label and rel==d['rel']}
        return ('explicit',hits.pop()) if len(hits)==1 else ('short',None)
    return 'context',None

def main():
    cfg=read('contract.json')
    for f,sha in cfg['pins'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==sha,f
    model,tok=load(os.environ['HYBRID_QWEN_WEIGHTS']);lm=model.language_model;rank=read('ranking.json');schema=read('decoding_schema.json')
    def ids(s):return tok.encode(s,add_special_tokens=False)
    def shell(s):
        return tok.apply_chat_template([dict(role='user',content=s)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    def split(facts,q):
        text=shell('Use these hypothetical facts and general knowledge. Give only the short answer.\n'+'\n'.join(facts)+'\nSPLITPOINT'+q)
        first,last=text.split('SPLITPOINT');return ids(first),ids(last)
    def run(tokens,cache,readout):
        hidden=model.model(mx.array([tokens]),cache=cache)
        if readout:
            hs=hidden[:,-1:];scores=lm.model.embed_tokens.as_linear(hs) if lm.args.tie_word_embeddings else lm.lm_head(hs)
            return scores[0,0].astype(mx.float32)
        mx.eval(hidden);return None
    def decode(caches,tokens,scale):
        outputs=[];first=None;positions=0
        for _ in range(24):
            values=[run(tokens,cache,True) for cache in caches];positions+=len(tokens)*len(caches)
            if len(values)>1:
                scores=values[0]+scale*(values[0]-values[1])
                scores=mx.where(values[0]>=mx.max(values[0])+np.log(.1),scores,-float('inf'))
            else:scores=values[0]
            token=int(mx.argmax(scores).item())
            if first is None:first=token
            if token in tok.eos_token_ids:break
            outputs.append(token)
            if '\n' in tok.decode(outputs):break
            tokens=[token]
        mx.synchronize()
        return tok.decode(outputs).strip().strip('*` ').split('\n')[0].rstrip('.'),positions
    modes=cfg['modes'];file=p/'fresh_rows.jsonl';done=set()
    if file.exists():
        prior=[json.loads(line) for line in file.read_text().splitlines()]
        done={x['id'] for x in prior}
        for uid in done:assert {x['mode'] for x in prior if x['id']==uid}==set(modes)
    start=time.perf_counter()
    with file.open('a') as out:
        for bi,b in enumerate(json.loads(gzip.decompress((p/'banks.json.gz').read_bytes()))):
            for index,item in enumerate(b['queries']):
                uid=item['id'];q=item['question']
                if uid in done:continue
                selections=rank[uid]
                order=modes.copy();random.Random(uid).shuffle(order);rows=[]
                for mode in order:
                    begin=time.perf_counter();typ,ans=route(q,b['documents'],schema) if mode!='all_context' else ('context',None)
                    positions=0
                    if typ=='short':
                        c=make_prompt_cache(model);ans,positions=decode([c],ids(shell('Answer with only the short factual answer.\n'+q)),0)
                    elif typ=='context':
                        group=selections['dense4' if mode=='dense' else 'adaptive'];ds=[b['documents'][i] for i in group]
                        current=[d['text']+'.' for d in ds];old=[d['old_text']+'.' for d in ds]
                        if mode=='paired':
                            current=['Previously: '+a+' Current: '+b for a,b in zip(old,current,strict=True)]
                        pre,tail=split(current,q);c=make_prompt_cache(model);run(pre,c,False);positions+=len(pre)
                        cs=[c]
                        if mode in ('native','whole'):
                            op,ot=split(old,q);assert ot==tail
                            oc=make_prompt_cache(model);run(op,oc,False);positions+=len(op)
                            if mode=='native':
                                # Shared attention address and convolution; contrast only cached recurrent matrices, before the question.
                                cf=copy.deepcopy(c)
                                for i,layer in enumerate(model.model.layers):
                                    if layer.is_linear:cf[i][1]=oc[i][1]
                            else:cf=oc
                            cs.append(cf)
                        ans,used=decode(cs,tail,1.);positions+=used
                    rows.append(dict(id=uid,bank=bi,mode=mode,route=typ,answer=ans,forward_positions=positions,elapsed_s=time.perf_counter()-begin))
                for x in rows:out.write(json.dumps(x,ensure_ascii=False)+'\n')
                out.flush()
                if index%12==11:print(bi,index+1,round(time.perf_counter()-start,1),flush=True)
    (p/'fresh_receipt.json').write_text(json.dumps(dict(cases=len(rank),modes=modes,wall_s=time.perf_counter()-start,prior=len(done)))+'\n')
if __name__=='__main__':main()
