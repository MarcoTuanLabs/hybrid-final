"""Measure how masking a proposed span in one report changes its conditional answer likelihood."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import mlx.core as mx
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'hoh_inverse'))
from query import Engine,history
from likelihood import likelihood

def swap_matrices(engine,active,neutral):
    mixed=copy.deepcopy(active)
    for j,layer in enumerate(engine.model.model.layers):
        if layer.is_linear:
            mixed[j][1]=neutral[j][1]
    return mixed

def measure(engine,text,doc,question,test=False):
    span=doc['span'];l,r=doc['start'],doc['end']
    assert span.strip() and text[l:r]==span
    def context(statement):
        return 'Quote the shortest exact phrase in the user report that answers the question, if one is present.\nDated user report: '+statement+'\nQuestion: '+question
    raw,tail=history(engine,context(text))
    masked=text[:l]+'[omitted]'+text[r:]
    erased,other=history(engine,context(masked))
    assert tail==other
    restored=swap_matrices(engine,raw,erased)
    if test:
        checked=likelihood(engine,swap_matrices(engine,raw,raw),tail,span)
    active=likelihood(engine,raw,tail,span)
    whole=likelihood(engine,erased,tail,span)
    matrix=likelihood(engine,restored,tail,span)
    if test:assert checked==active,(checked,active)
    return dict(active=active,whole_gain=active-whole,matrix_gain=active-matrix)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('model',type=Path);a=ap.parse_args()
    rows=list(map(json.loads,(HERE/'retrieval.jsonl').read_text().splitlines()));eng=Engine(a.model);mx.set_cache_limit(64*1024**2)
    dest=HERE/'supports.jsonl';seen={x['id'] for x in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    start=time.monotonic()
    with dest.open('a') as out:
        for row in rows:
            if row['id'] in seen:continue
            tick=time.monotonic();n=eng.positions;values={}
            for j,k in enumerate(row['support']):
                doc=row['docs'][k];values[k]=measure(eng,doc['text'],doc,row['form'],j==0)
                mx.clear_cache()
            out.write(json.dumps(dict(id=row['id'],scores=values,positions=eng.positions-n,seconds=time.monotonic()-tick,identical_matrix_control=True))+'\n');out.flush()
            print(len(seen)+1,round(time.monotonic()-start,1),flush=True);seen.add(row['id'])
            if time.monotonic()-start>2400:raise TimeoutError('Support phase budget')
if __name__=='__main__':main()
