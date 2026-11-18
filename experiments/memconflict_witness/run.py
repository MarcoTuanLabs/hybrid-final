"""Compare dated dialog readers and a recurrent intervention on source attribution."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from query import Engine,history,generate

DIRECT='Answer the question about this user from the dated conversation excerpts. Give a complete, concise factual answer in one line.'
TYPED=(DIRECT+' If asked what changed, give the prior and new values in order. For a preference, state its specific condition. '
       'If stable facts conflict, identify the best supported fact and mention the competing value. Separate the user from people they discuss and distinguish user statements from assistant suggestions.')

def prompt(q,records,instruction,flip=False):
    parts=[instruction]
    for row in records:
        lines=[row['date']]
        for turn in row['turns']:
            who=turn['speaker']
            if flip:who='assistant' if who=='user' else 'user'
            lines.append(who+': '+turn['text'])
        parts.append('\n'.join(lines))
    parts.append('As of '+q['date']+'\nQuestion: '+q['question'])
    return '\n\n'.join(parts)

def contrasted(engine,start,control,kind,check=False):
    positive,tail=history(engine,start);negative,alternate=history(engine,control)
    assert tail==alternate
    if kind=='recurrent':
        for j,layer in enumerate(engine.model.model.layers):
            if layer.is_linear:
                swapped_matrix=negative[j][1]
                negative[j]=copy.deepcopy(positive[j])
                negative[j][1]=swapped_matrix
            else:
                negative[j]=copy.deepcopy(positive[j])
    ids=tail;out=[]
    for t in range(80):
        base=engine.forward(ids,positive)
        if t<8:
            other=engine.forward(ids,negative);joint=base+.75*(base-other)
            chosen=mx.argmax(mx.where(base>=mx.max(base)+mx.log(mx.array(.05)),joint,-float('inf')))
            if check:assert int(chosen.item())==int(mx.argmax(base).item())
            if t==7:del negative
        else:chosen=mx.argmax(base)
        token=int(chosen.item())
        if token in engine.tokenizer.eos_token_ids:break
        out.append(token)
        if '\n' in engine.tokenizer.decode(out):break
        ids=[token]
    mx.synchronize()
    return engine.tokenizer.decode(out).split('\n',1)[0].strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',type=Path);a=parser.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    qs=json.loads((HERE/'questions.json').read_text());ev=json.loads((HERE/'evidence.json').read_text())
    dest=HERE/'answers.jsonl';seen={r['id'] for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    start=time.monotonic()
    with dest.open('a') as f:
        for ordinal,q in enumerate(qs):
            uid=q['id']
            if uid in seen:continue
            tick=time.monotonic();pos=engine.positions;docs=ev[uid];answers={};cache={}
            def plain(text):
                if text not in cache:cache[text]=generate(engine,text,80)
                return cache[text]
            for arm in ['standard','diverse','witness','wide']:
                answers[arm]=plain(prompt(q,docs[arm],DIRECT))
            for arm in ['wide','witness']:
                answers[arm+'_typed']=plain(prompt(q,docs[arm],TYPED))
            real=prompt(q,docs['witness'],TYPED)
            critique=(real+'\nA first reader answered: '+answers['witness_typed']+
                      '\nRecheck the dates, user and competing evidence. Return the best complete one-line answer.')
            answers['ordinary_critique']=plain(critique)
            false=prompt(q,docs['witness'],TYPED,True)
            for variant in ['whole','recurrent']:
                answers[variant]=contrasted(engine,real,false,variant)
            neutral=None
            if ordinal==0:
                neutral=contrasted(engine,real,real,'recurrent',True)
                assert neutral==answers['witness_typed'],(neutral,answers['witness_typed'])
            elapsed=time.monotonic()-tick
            f.write(json.dumps(dict(id=uid,answers=answers,identical_evidence_control=neutral,ordinary_calls=len(cache),positions=engine.positions-pos,seconds=elapsed),ensure_ascii=False)+'\n');f.flush()
            print(ordinal+1,round(elapsed,1),flush=True);mx.clear_cache()
            if time.monotonic()-start>1800:raise TimeoutError('Answer generation budget')
if __name__=='__main__':main()
