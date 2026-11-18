"""Answer from available dialogue boundaries and an explicit stable-attribute baseline."""
import argparse
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from query import Engine,generate

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',type=Path);a=parser.parse_args()
    load=lambda n:json.loads((HERE/(n+'.json')).read_text())
    qs=load('questions');hs=load('histories');ev=load('evidence');mx.set_cache_limit(64*1024**2);engine=Engine(a.model);start=time.monotonic()
    dest=HERE/'boundary_answers.jsonl'
    seen={r['id'] for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    with dest.open('a') as f:
        for q in qs:
            uid=q['id']
            if uid in seen:continue
            t=time.monotonic();n=engine.positions;h=hs[uid];parts=[]
            for j,s in enumerate(h):
                msgs=[v['text'] for v in s['turns'] if v['speaker']=='user']
                if msgs:parts.append(s['date']+' USER: '+'\n'.join(msgs if j==len(h)-1 else msgs[:1]))
            instruction=("Answer about the user from the dated conversations. For a changing state, give the earlier and updated values; for preferences report the condition. "
                         "For a stable biographical fact, if reports disagree cite both and prefer the earliest firsthand report unless the conflict is resolved independently. Distinguish other people. Give just the concise complete answer.")
            history='\n'.join(parts)
            witness='\n'.join(row['date']+' '+' '.join(v['speaker']+': '+v['text'] for v in row['turns']) for row in ev[uid]['witness'])
            ans={}
            for name,extra in [('boundaries',''),('boundaries_plus_witness','\nOther retrieved excerpts:\n'+witness)]:
                prompt=instruction+'\nDated user reports:\n'+history+extra+'\nAs of '+q['date']+'\nQuestion: '+q['question']
                ans[name]=generate(engine,prompt,80)
            f.write(json.dumps(dict(id=uid,answers=ans,positions=engine.positions-n,seconds=time.monotonic()-t),ensure_ascii=False)+'\n');f.flush();mx.clear_cache()
            if time.monotonic()-start>900:raise TimeoutError('Boundary-control budget')
if __name__=='__main__':main()
