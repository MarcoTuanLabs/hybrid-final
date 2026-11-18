"""Estimate matrix arithmetic from the model shapes and each measured execution."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent

def decoder(cfg,n,offset,outputs):
    h=cfg['hidden_size']
    full=cfg['num_hidden_layers']//cfg['full_attention_interval']
    linear=cfg['num_hidden_layers']-full
    dk=cfg['linear_num_key_heads']*cfg['linear_key_head_dim']
    dv=cfg['linear_num_value_heads']*cfg['linear_value_head_dim']
    qa=cfg['num_attention_heads']*cfg['head_dim']
    kv=cfg['num_key_value_heads']*cfg['head_dim']
    recurrent=h*(2*dk+3*dv+2*cfg['linear_num_value_heads'])
    recurrent+=cfg['linear_conv_kernel_dim']*(2*dk+dv)+4*dv*cfg['linear_key_head_dim']
    body=3*h*cfg['intermediate_size']*cfg['num_hidden_layers']+linear*recurrent+full*h*(3*qa+2*kv)
    return n*body+2*full*qa*(offset*n+n*(n+1)//2)+outputs*h*cfg['vocab_size']

def encoder(n):
    h=768
    return 12*(n*(4*h*h+2*h*3072)+2*n*n*h)+2*n*h

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('tag',choices=['dev','test'])
    p.add_argument('inputs',type=Path);p.add_argument('qwen',type=Path)
    a=p.parse_args();inputs={r['id']:r for r in json.loads(a.inputs.read_text())}
    cfg=json.loads((a.qwen/'config.json').read_text())
    cfg=cfg.get('text_config',cfg)
    tokenizer=AutoTokenizer.from_pretrained(a.qwen,local_files_only=True)
    def tokens(row,field):
        text=('Answer the question about this Wikipedia page. Give only the shortest factual answer.\n'
              +'Page: '+row['title']+'\nQuestion: '+row['question']+'\nEvidence:\n'+row[field])
        formatted=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,
                                                add_generation_prompt=True,enable_thinking=False)
        return tokenizer.encode(formatted,add_special_tokens=False)
    results={}
    for model in ['cpu','qwen']:
        rows=list(map(json.loads,(HERE/(a.tag+'_'+model+'.jsonl')).read_text().splitlines()))
        macs={}
        for row in rows:
            source=inputs[row['id']]
            if model=='qwen':
                old=tokens(source,'old');new=tokens(source,'current')
                common=next((i for i,(x,y) in enumerate(zip(old,new)) if x!=y),min(len(old),len(new)))
                checkpoints=[min(pos,len(old)-1) for pos in range(32,len(old)-1+32,32)]
                split=max((c for c in checkpoints if c<=min(common,len(new)-1)),default=0)
            for mode,run in row['policies'].items():
                if mode=='guarded_native':
                    continue  # The attention trace performs extra operations beyond the base decoder.
                n=run['old_positions'];extra=run['total_positions']-n
                if model=='cpu':
                    value=encoder(n)+encoder(extra)
                else:
                    outs=n-len(old)+1
                    assert 1<=outs<=32,(n,len(old),outs)
                    value=decoder(cfg,n,0,outs)
                    if extra:
                        out2=extra-(len(new)-split)+1
                        assert 1<=out2<=32,(extra,len(new),split,out2)
                        value+=decoder(cfg,extra,split,out2)
                macs[mode]=macs.get(mode,0)+value
        base='full' if model=='cpu' else 'prefix'
        results[model]={m:dict(dense_equivalent_ops=2*n,saving_vs_reread=1-n/macs[base]) for m,n in macs.items()}
    (HERE/(a.tag+'_arithmetic.json')).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
