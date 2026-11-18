"""A no-change edition pair must give zero whole-cache and recurrent-only contrasts."""
import argparse
import json
from pathlib import Path
import sys
import mlx.core as mx
from likelihood import score,Engine,SOURCE

HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);a=p.parse_args()
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model)
    qs=json.loads((SOURCE/'questions.json').read_text());pool=json.loads((HERE/'pool.json').read_text())
    checks=[]
    for j in [0,15,31]:
        q=qs[j];row=dict(pool[q['id']][0]);row['old']=row['current']
        out=score(engine,row,q['question'])
        assert abs(out['whole_gain'])<=1e-5 and abs(out['recurrent_gain'])<=1e-5,out
        checks.append(dict(id=q['id'],whole=out['whole_gain'],recurrent=out['recurrent_gain']))
        mx.clear_cache()
    (HERE/'null_control.json').write_text(json.dumps(checks,indent=2)+'\n')
if __name__=='__main__':main()
