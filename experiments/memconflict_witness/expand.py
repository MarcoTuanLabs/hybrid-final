"""Propose dialogue retrieval terms from the incoming question alone."""
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
    mx.set_cache_limit(64*1024**2);engine=Engine(a.model);out={};tick=time.monotonic()
    for q in json.loads((HERE/'questions.json').read_text()):
        start=time.monotonic();pos=engine.positions
        prompt="List short search phrases and synonyms likely to occur in a person's actual conversation about this question. Avoid inventing names, values or answers. Only give search terms.\nQuestion: "+q['question']
        out[q['id']]=dict(rewrite=generate(engine,prompt,50),positions=engine.positions-pos,seconds=time.monotonic()-start)
        mx.clear_cache()
        if time.monotonic()-tick>1800:raise TimeoutError('Expansion phase')
    (HERE/'expansions.json').write_text(json.dumps(out,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
