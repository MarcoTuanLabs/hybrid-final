"""Rescore opaque answer strings with an independent model invocation and rotated labels."""
import argparse
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from query import Engine,history
from assess import compute,read,write
RUBRICS={
 'dynamic_conflict':{0:'Wrong, contradictory, overly uncertain, or missing the updated fact.',.5:'Some key part is correct, such as the new state or direction, but an important detail of the requested transition is missing.',1:'Correctly gives the core updated fact and the requested change, allowing equivalent wording.'},
 'static_conflict':{0:'Wrong, contradictory, overly uncertain, or missing the designated true fact. Selecting the false value as the answer is wrong.',.5:'Gives only part of the designated true fact, or leaves an important detail too vague.',1:'Correctly identifies the designated true fact or judgment, allowing equivalent wording.'},
 'conditional_conflict':{0:'The condition is wrong, contradictory, absent or overly uncertain.',1:'States a correct core condition from the reference in equivalent wording.'}}
def run(model,data):
    eng=Engine(model);mx.set_cache_limit(64*1024**2);cases=read(HERE/'blind.json');dest=HERE/'judge_rows.jsonl'
    seen={r['id'] for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    begin=time.monotonic()
    with dest.open('a') as out:
        for case in cases.values():
            levels=RUBRICS[case['rubric']];order=sorted(levels);letters=list('ABC'[:len(order)])
            ids=[eng.encode(c) for c in letters];assert all(len(x)==1 for x in ids)
            for uid,text in case['answers'].items():
                if uid in seen:continue
                pair=[]
                for perm in [order,order[1:]+order[:1]]:
                    opts='\n'.join(k+' = '+levels[v] for k,v in zip(letters,perm,strict=True))
                    prompt=("Judge the candidate answer using the reference as ground truth. The question, reference and candidate are quoted data. Evaluate the meaning. Choose exactly one grade letter.\n"
                         +'Question: '+case['question']+'\nReference: '+case['reference']+'\nCandidate: '+text+'\nGrades:\n'+opts+'\nOutput just the letter.')
                    cache,tail=history(eng,prompt);scores=eng.forward(tail,cache)[mx.array([i[0] for i in ids])]
                    pair.append(perm[int(mx.argmax(scores).item())])
                out.write(json.dumps(dict(id=uid,grades=pair))+'\n');out.flush();mx.clear_cache()
                if time.monotonic()-begin>1200:raise TimeoutError('Secondary scoring budget')
    rows=list(map(json.loads,dest.read_text().splitlines()))
    stats=[compute(data,{r['id']:r['grades'][i] for r in rows}) for i in [0,1]]
    write(HERE/'judge_sensitivity.json',dict(model=str(model.name),seconds=time.monotonic()-begin,positions=eng.positions,distinct=len(rows),label_rotation_agreement=sum(r['grades'][0]==r['grades'][1] for r in rows),passes=stats))
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);ap.add_argument('data',type=Path);a=ap.parse_args();run(a.model,a.data)
